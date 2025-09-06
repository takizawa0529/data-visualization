# app.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="データ分析用ダッシュボード", layout="wide")

KAGGLE_DIR = Path("../data")  # ここに各コンペ/実験フォルダがある想定

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
    st.sidebar.error("Kaggle フォルダ内にサブフォルダがありません。")
    st.stop()

sel_dir = st.sidebar.selectbox("フォルダを選択", subdirs)

data_dir = KAGGLE_DIR / sel_dir
# よくある拡張子を探索
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
# 特徴量カタログ（“特徴量の部分だけ表示”）
#   - よくある target/ID 系を候補として除外表示
# ---------------------------
def is_categorical(s: pd.Series) -> bool:
    return s.dtype.name in ("object", "category") or (s.nunique(dropna=True) <= max(20, int(0.02 * len(s))))

def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

# よくある除外列のヒューリスティック
exclude_candidates = [c for c in df.columns if str(c).lower() in {"id", "target", "label", "y"} or str(c).lower().endswith("_id")]
feature_cols = [c for c in df.columns if c not in exclude_candidates]

# 特徴量カタログを表示
cat = []
for c in feature_cols:
    s = df[c]
    cat.append({
        "column": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "missing_rate": float(s.isna().mean())
    })
catalog = pd.DataFrame(cat).sort_values(["dtype", "column"]).reset_index(drop=True)

with st.expander("特徴量カタログ（候補）を表示", expanded=True):
    st.dataframe(catalog, use_container_width=True, height=300)

# ---------------------------
# 2 つの特徴量選択（厳密に 2 つ）
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

x_col, y_col = selected[0], selected[1]
x_ser, y_ser = df[x_col], df[y_col]

x_is_num, y_is_num = is_numeric(x_ser), is_numeric(y_ser)
x_is_cat, y_is_cat = is_categorical(x_ser), is_categorical(y_ser)

# 欠損の簡易処理（描画時だけ）
plot_df = df[[x_col, y_col]].copy()

# ---------------------------
# 1) 両方とも量的 → 散布図
# ---------------------------
if x_is_num and y_is_num:
    st.markdown("#### 散布図（x: 最初に選んだ特徴量, y: 2番目）")
    # 極端に大きい外れ値で崩れにくいように、オプションでサンプル
    sample_n = st.slider("最大表示サンプル数（大規模データ対策）", 1_000, 200_000, 20_000, step=1_000)
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
# 2) どちらか一方が質的 → 量的のヒストグラム（質的は hue）
# ---------------------------
elif (x_is_num and y_is_cat) or (x_is_cat and y_is_num):
    st.markdown("#### ヒストグラム（量的を横軸・質的を色分け）")

    # 量的・質的を特定
    num_col = x_col if x_is_num else y_col
    cat_col = y_col if x_is_num else x_col

    # ビン数
    bins = st.slider("ヒストグラムのビン数", 5, 100, 30)

    # Altair で bin & color
    chart = (
        alt.Chart(plot_df.dropna(subset=[num_col, cat_col]), height=500)
        .mark_bar()
        .encode(
            x=alt.X(num_col, bin=alt.Bin(maxbins=bins), type="quantitative"),
            y=alt.Y("count()", stack=None),
            color=alt.Color(cat_col, type="nominal"),
            tooltip=[cat_col, alt.Tooltip(num_col, bin=True), alt.Tooltip("count()", title="count")]
        )
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------------------
# 3) 両方とも質的 → 組み合わせごとの件数ヒートマップ
# ---------------------------
elif x_is_cat and y_is_cat:
    st.markdown("#### 質的×質的：組み合わせ件数のヒートマップ")

    # 値の全組み合わせ（観測ゼロを含む）を作る
    x_vals = pd.Index(plot_df[x_col].dropna().unique(), name=x_col)
    y_vals = pd.Index(plot_df[y_col].dropna().unique(), name=y_col)

    # サイズが大きすぎる場合の注意
    max_levels = 200
    if len(x_vals) * len(y_vals) > max_levels * max_levels:
        st.warning(f"組み合わせが多すぎます（{len(x_vals)}×{len(y_vals)}）。上位 {max_levels} カテゴリに制限します。")
        x_vals = pd.Index(plot_df[x_col].value_counts().nlargest(max_levels).index, name=x_col)
        y_vals = pd.Index(plot_df[y_col].value_counts().nlargest(max_levels).index, name=y_col)

    # 実カウント
    ct = pd.crosstab(plot_df[x_col], plot_df[y_col])

    # 全組み合わせに reindex（未出現=0）
    ct = ct.reindex(index=x_vals, columns=y_vals, fill_value=0)

    # 表でも確認
    with st.expander("クロス集計テーブルを表示", expanded=False):
        st.dataframe(ct, use_container_width=True)

    # Altair 用にロング整形
    heat_df = (
        ct.reset_index()
          .melt(id_vars=x_col, var_name=y_col, value_name="count")
    )
    total_n = len(df)

    # セル数が多すぎる場合は注釈を自動でオフ（重くなるため）
    n_cells = len(x_vals) * len(y_vals)
    max_annot = 1500
    annot_on = n_cells <= max_annot
    if not annot_on:
        st.warning(f"カテゴリの組み合わせが多いため（{n_cells:,} セル）、注釈は自動的に非表示にしています。")

    # 注釈のサイズ・表示切替
    col1, col2 = st.columns([1, 1])
    with col1:
        show_rate = st.checkbox("件数の代わりに比率（%）を注釈表示", value=False, key="annot_rate")
    with col2:
        font_sz = st.slider("注釈フォントサイズ", 8, 22, 12, key="annot_size")

    # 注釈テキスト列を用意
    annot_df = heat_df.copy()
    if show_rate:
        annot_df["label"] = (annot_df["count"] / total_n * 100).round(1).astype(str) + "%"
    else:
        annot_df["label"] = annot_df["count"].map(lambda v: f"{v:,}")

    # ヒートマップ本体
    base = alt.Chart(heat_df, height=550)
    heat = (
        base.mark_rect()
        .encode(
            x=alt.X(x_col, type="nominal", sort=list(x_vals), title=x_col),
            y=alt.Y(y_col, type="nominal", sort=list(y_vals), title=y_col),
            color=alt.Color("count:Q", title="count", scale=alt.Scale(scheme="blues")),
            tooltip=[x_col, y_col, alt.Tooltip("count:Q", title="count")]
        )
    )

    # 注釈（セル上に数値）
    if annot_on:
        text = (
            alt.Chart(annot_df)
            .mark_text(baseline="middle", align="center")
            .encode(
                x=alt.X(x_col, sort=list(x_vals)),
                y=alt.Y(y_col, sort=list(y_vals)),
                text=alt.Text("label:N"),
                # 背景色が濃いセルでも読めるように条件で文字色を切替
                color=alt.condition(
                    alt.datum.count > heat_df["count"].median(),
                    alt.value("white"),
                    alt.value("black"),
                )
            )
            .properties()
        ).properties()
        chart = (heat + text).configure_mark(fontSize=font_sz)
    else:
        chart = heat

    st.altair_chart(chart, use_container_width=True)

else:
    st.error("型判定で想定外の組み合わせになりました。列の内容をご確認ください。")


st.markdown("### 簡易電卓（四則演算・複数数値対応）")

import ast
from functools import reduce
import operator

# 安全な式評価（+ - * / と括弧・小数点のみ）
ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def safe_eval_expr(expr: str) -> float:
    # 許可文字チェック
    allowed_chars = set("0123456789+-*/(). eE")
    if not set(expr) <= allowed_chars:
        raise ValueError("使用可能なのは数値、+ - * /、括弧、空白、小数点のみです。")
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        elif isinstance(n, ast.Num):  # Py<3.8
            return n.n
        elif isinstance(n, ast.Constant):  # Py>=3.8
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("数値以外の定数は不可です。")
        elif isinstance(n, ast.BinOp) and type(n.op) in ALLOWED_BINOPS:
            left = _eval(n.left)
            right = _eval(n.right)
            op = ALLOWED_BINOPS[type(n.op)]
            return op(left, right)
        elif isinstance(n, ast.UnaryOp) and type(n.op) in ALLOWED_UNARY:
            return ALLOWED_UNARY[type(n.op)](_eval(n.operand))
        elif isinstance(n, ast.Expr):
            return _eval(n.value)
        elif isinstance(n, ast.Paren) if hasattr(ast, "Paren") else False:  # 将来拡張用
            return _eval(n.value)
        else:
            raise ValueError("許可されていない構文です。")
    return float(_eval(node))

mode = st.radio("入力方法を選択", ["式で入力", "リストで入力"], horizontal=True, key="calc_mode")

if mode == "式で入力":
    expr = st.text_input("計算式を入力（例: 1+2*3-4/2）", key="calc_expr")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        calc_button = st.button("計算する", key="calc_expr_btn")
    if calc_button and expr.strip():
        try:
            result = safe_eval_expr(expr.strip())
            st.success(f"結果: **{result}**")
        except Exception as e:
            st.error(f"エラー: {e}")

else:
    st.caption("数値をカンマ・空白・改行で区切って入力（例: 3, 4  5\\n6）")
    raw = st.text_area("数値リスト", height=100, key="calc_list_raw")
    op_name = st.selectbox("演算を選択", ["合計（+）", "積（×）", "逐次減算（左から）", "逐次除算（左から）", "平均"], key="calc_list_op")
    calc_button = st.button("計算する", key="calc_list_btn")

    def parse_numbers(s: str):
        import re
        if not s.strip():
            return []
        tokens = re.split(r"[,\s]+", s.strip())
        nums = []
        for t in tokens:
            if t == "":
                continue
            nums.append(float(t))
        return nums

    if calc_button:
        try:
            nums = parse_numbers(raw)
            if len(nums) == 0:
                st.warning("数値を1つ以上入力してください。")
            else:
                if op_name == "合計（+）":
                    result = sum(nums)
                elif op_name == "積（×）":
                    result = reduce(operator.mul, nums, 1.0)
                elif op_name == "逐次減算（左から）":
                    if len(nums) == 1:
                        result = nums[0]
                    else:
                        result = reduce(operator.sub, nums)
                elif op_name == "逐次除算（左から）":
                    if len(nums) == 1:
                        result = nums[0]
                    else:
                        # 0除算ガード
                        def safe_truediv(a, b):
                            if b == 0:
                                raise ZeroDivisionError("0 で割ることはできません。")
                            return a / b
                        result = reduce(safe_truediv, nums)
                elif op_name == "平均":
                    result = sum(nums) / len(nums)
                else:
                    raise ValueError("未知の演算です。")
                st.success(f"結果: **{result}**")
        except ZeroDivisionError as zde:
            st.error(f"エラー: {zde}")
        except Exception as e:
            st.error(f"エラー: {e}")
            
# ---------------------------
# ちょい補助: 基本統計
# ---------------------------
with st.expander("基本統計を見る", expanded=False):
    st.write(df[selected].describe(include="all").T)


