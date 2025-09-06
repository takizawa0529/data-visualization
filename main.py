import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from functools import reduce
import operator

st.set_page_config(page_title="データ分析用ダッシュボード", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
KAGGLE_DIR = BASE_DIR / "competition"

st.title("データ分析用ダッシュボード")

# ---------------------------
# サイドバー: フォルダ & ファイル選択
# ---------------------------
st.sidebar.header("データ選択")

if not KAGGLE_DIR.exists():
    st.sidebar.error(f"'{KAGGLE_DIR}' フォルダが見つかりません。プロジェクト直下に作成してください。")
    st.stop()

subdirs = sorted([d.name for d in KAGGLE_DIR.iterdir() if d.is_dir()])
if not subdirs:
    st.sidebar.error("competition フォルダ内にサブフォルダがありません。")
    st.stop()

sel_dir = st.sidebar.selectbox("フォルダを選択", subdirs)
data_dir = KAGGLE_DIR / sel_dir

# データファイル候補
candidate_files = sorted([p for p in data_dir.glob("**/*") if p.suffix.lower() in [".csv", ".parquet", ".feather"]])
if not candidate_files:
    st.sidebar.error("選んだフォルダに .csv / .parquet / .feather が見つかりません。")
    st.stop()

sel_file = st.sidebar.selectbox("データファイルを選択", [str(p.relative_to(KAGGLE_DIR)) for p in candidate_files])

@st.cache_data(show_spinner=True)
def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    raise ValueError("未対応の拡張子です")

df = load_df(KAGGLE_DIR / sel_file)
st.caption(f"読み込み: `{sel_file}`  /  形状: {df.shape[0]:,} 行 × {df.shape[1]:,} 列")

# ---------------------------
# 特徴量カタログ
# ---------------------------
def is_categorical(s: pd.Series) -> bool:
    return s.dtype.name in ("object", "category") or (s.nunique(dropna=True) <= max(20, int(0.02 * len(s))))

def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

exclude_candidates = [c for c in df.columns if str(c).lower() in {"id", "target", "label", "y"} or str(c).lower().endswith("_id")]
feature_cols = [c for c in df.columns if c not in exclude_candidates]

catalog = pd.DataFrame([{
    "column": c,
    "dtype": str(df[c].dtype),
    "n_unique": int(df[c].nunique(dropna=True)),
    "missing_rate": float(df[c].isna().mean())
} for c in feature_cols]).sort_values(["dtype", "column"]).reset_index(drop=True)

with st.expander("特徴量カタログ（候補）を表示", expanded=True):
    st.dataframe(catalog, use_container_width=True, height=300)

# ---------------------------
# 特徴量選択
# ---------------------------
st.subheader("可視化する特徴量を2つ選択")
selected = st.multiselect(
    "2つだけ選んでください（順番はそのまま x→y に反映）",
    options=feature_cols,
    max_selections=2
)

if len(selected) != 2:
    st.info("特徴量をちょうど2つ選択すると、下に可視化が表示されます。")
    st.stop()

x_col, y_col = selected
x_ser, y_ser = df[x_col], df[y_col]
x_is_num, y_is_num = is_numeric(x_ser), is_numeric(y_ser)
x_is_cat, y_is_cat = is_categorical(x_ser), is_categorical(y_ser)

plot_df = df[[x_col, y_col]].copy()

# ---------------------------
# 1) 数値×数値 → 散布図
# ---------------------------
if x_is_num and y_is_num:
    st.markdown("#### 散布図")
    sample_n = st.slider("最大表示サンプル数", 1_000, 200_000, 20_000, step=1_000)
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    chart = (
        alt.Chart(plot_df.dropna(), height=500)
        .mark_point(opacity=0.6)
        .encode(
            x=alt.X(x_col, type="quantitative"),
            y=alt.Y(y_col, type="quantitative"),
            tooltip=[x_col, y_col]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------------------
# 2) 数値×カテゴリ → ヒストグラム
# ---------------------------
elif (x_is_num and y_is_cat) or (x_is_cat and y_is_num):
    st.markdown("#### ヒストグラム")
    num_col = x_col if x_is_num else y_col
    cat_col = y_col if x_is_num else x_col
    bins = st.slider("ヒストグラムのビン数", 5, 100, 30)

    chart = (
        alt.Chart(plot_df.dropna(subset=[num_col, cat_col]), height=500)
        .mark_bar()
        .encode(
            x=alt.X(num_col, bin=alt.Bin(maxbins=bins)),
            y="count()",
            color=cat_col,
            tooltip=[cat_col, alt.Tooltip(num_col, bin=True), alt.Tooltip("count()", title="count")]
        )
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------------------
# 3) カテゴリ×カテゴリ → ヒートマップ
# ---------------------------
elif x_is_cat and y_is_cat:
    st.markdown("#### ヒートマップ")

    ct = pd.crosstab(plot_df[x_col], plot_df[y_col])
    x_vals = ct.index
    y_vals = ct.columns
    heat_df = ct.reset_index().melt(id_vars=x_col, var_name=y_col, value_name="count")

    # 描画方法を切替（Altair or Matplotlib）
    mode = st.radio("描画方法を選択", ["Altair（インタラクティブ・PC推奨）", "Matplotlib（モバイル推奨）"], horizontal=True)

    if mode.startswith("Altair"):
        base = alt.Chart(heat_df, height=550)
        heat = base.mark_rect().encode(
            x=alt.X(x_col, type="nominal", sort=list(x_vals)),
            y=alt.Y(y_col, type="nominal", sort=list(y_vals)),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[x_col, y_col, "count:Q"]
        )
        st.altair_chart(heat, use_container_width=True)

    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# ---------------------------
# 基本統計
# ---------------------------
with st.expander("基本統計を見る", expanded=False):
    st.write(df[selected].describe(include="all").T)

# ---------------------------
# 簡易電卓
# ---------------------------
st.markdown("### 簡易電卓")

ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def safe_eval_expr(expr: str) -> float:
    allowed_chars = set("0123456789+-*/(). eE")
    if not set(expr) <= allowed_chars:
        raise ValueError("使用可能なのは数値、+ - * /、括弧、小数点のみです。")
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        if isinstance(n, ast.Constant): return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ALLOWED_UNARY:
            return ALLOWED_UNARY[type(n.op)](_eval(n.operand))
        raise ValueError("許可されていない式です")
    return _eval(node)

mode = st.radio("入力方法", ["式で入力", "リストで入力"], horizontal=True)
if mode == "式で入力":
    expr = st.text_input("計算式を入力（例: 1+2*3-4/2）")
    if st.button("計算する") and expr.strip():
        try:
            result = safe_eval_expr(expr.strip())
            st.success(f"結果: **{result}**")
        except Exception as e:
            st.error(f"エラー: {e}")
else:
    raw = st.text_area("数値リスト（カンマ・空白・改行区切り）", height=100)
    op_name = st.selectbox("演算を選択", ["合計（+）", "積（×）", "逐次減算", "逐次除算", "平均"])
    if st.button("計算する（リスト）"):
        try:
            nums = [float(x) for x in raw.replace(",", " ").split() if x]
            if not nums:
                st.warning("数値を入力してください。")
            else:
                if op_name == "合計（+）": result = sum(nums)
                elif op_name == "積（×）": result = reduce(operator.mul, nums, 1.0)
                elif op_name == "逐次減算": result = reduce(operator.sub, nums)
                elif op_name == "逐次除算":
                    def safe_div(a, b): return a / b if b != 0 else float("inf")
                    result = reduce(safe_div, nums)
                elif op_name == "平均": result = sum(nums)/len(nums)
                st.success(f"結果: **{result}**")
        except Exception as e:
            st.error(f"エラー: {e}")

            
# ---------------------------
# ちょい補助: 基本統計
# ---------------------------
with st.expander("基本統計を見る", expanded=False):
    st.write(df[selected].describe(include="all").T)

if (x_is_cat) & (y_is_num):
    with st.expander("クラスごとの基本統計を見る", expanded=False):
        st.write(df[selected].groupby(y_col).describe(include="all").T)
