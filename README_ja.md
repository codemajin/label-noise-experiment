# label-noise-experiment

[English](README.md) | 日本語

cleanlab（Confident Learning）と biquality-learn を、**人手を介さずに**組み合わせたときの効果を調べる実験。
結果は論文にまとめてある（英語版 `latex/main.pdf`、日本語版 `latex/main_ja.pdf`）。

## 実行

```sh
uv sync
uv run python -m label_noise_experiment.run --model lr    # 約 6 分（24 コア）
uv run python -m label_noise_experiment.run --model hgb   # 約 60 分
uv run python -m label_noise_experiment.summarize         # results/summary.md と図を出力

# 論文用の追加実験と表・図の書き出し
# （run を含む全実験は sh rerun_all.sh でまとめて実行できる。合計約 3 時間）
for m in lr hgb; do
  uv run python -m label_noise_experiment.oracle_check --model $m --oracle-frac 0.1
  uv run python -m label_noise_experiment.oracle_check --model $m --oracle-frac 0.3
  uv run python -m label_noise_experiment.mechanism --model $m
done
uv run python -m label_noise_experiment.hgb_lr_check      # HGB の学習率をテストデータを使わず確認
uv run python -m label_noise_experiment.export_latex      # 表（英・日）・図・本文の数値マクロ（latex/numbers.tex）
wsl sh latex/build.sh                                      # WSL の Docker（texlive/texlive）で main.pdf と main_ja.pdf をビルド
```

biquality-learn 0.1.0 は scikit-learn 1.6 以降で動かない（`_validate_data` が削除されたため）ので、`scikit-learn<1.6` に固定している。

## ファイル構成

| パス | 内容 |
|---|---|
| `src/label_noise_experiment/` | 実験コード（データセット、ノイズ、手法、実行・集計・書き出し） |
| `results/<model>/` | 実験結果の CSV（`results.csv`、`diagnostics.csv`、`oracle_check_{10,30}.csv`、`mechanism.csv`、`hgb/lr_check.csv`） |
| `results/summary.md` | 集計表と図 |
| `latex/main.tex`, `latex/main_ja.tex` | 論文（英語版・日本語版）。本文の数値は `numbers.tex`、表は `tables/{en,ja}/` を読み込む（いずれも `export_latex` が生成） |
| `review_response.md` | 論文へのレビュー指摘への対応 |

## 設定

| 項目 | 内容 |
|---|---|
| データセット | digits（1,797 件・10 クラス）、synth（make_classification・8,000 件・6 クラス）、covtype（12,000 件に層化抽出・7 クラス） |
| ノイズ | symmetric（一様ランダムな別クラスへ）、pair（クラスごとに決まった別クラスへ）、instance（クリーンラベルで学習したモデルが迷うサンプルほど、そのモデルが 2 番目に推すクラスへ）。いずれもノイズ率は 20% / 40% |
| 分割 | 学習 70% / テスト 30%。ノイズは学習側だけに入れ、テストラベルはクリーンのまま |
| ベースモデル | LogisticRegression（lr）、HistGradientBoosting（hgb、learning_rate=0.05） |
| 反復 | 5 シード。2 モデル × 90 条件 |
| cleanlab | 5 分割 CV の out-of-sample `pred_probs` を `find_label_issues` と `get_label_quality_scores` に渡す |

HGB の学習率 0.05 は、当初シード 0 のテストデータでの精度を見て決めた。テストデータを使わない学習データ内の交差検証（`hgb_lr_check`）でも同じ選択になることを確認している（covtype で 0.1 が 76.1%、0.05 が 79.1%）。

## 比較した手法

| # | 手法 | 内容 |
|---|---|---|
| 1 | `baseline` | ノイズ付きラベルでそのまま学習 |
| 2 | `cl_prune` | cleanlab がフラグを付けたサンプルを除去して一様重みで再学習 |
| 2 | `cl_cleanlearning` | cleanlab 標準の `CleanLearning`。除去するサンプルは `cl_prune` と同じで、クラスごとに 1/推定ノイズ行列の対角 の重みを付けて再学習 |
| 3 | `bq/<split>/<algo>` | cleanlab の出力から疑似 trusted/untrusted を作り、biquality-learn で学習 |
| 4 | `cl_weight/*` | ラベル品質スコア（self_confidence / normalized_margin）をサンプル重みにする |
| 4 | `cl_feature` | ラベル品質スコアを特徴量に追加して学習し、推論時はスコア=1（最も信頼できる）を入れる |
| ref | `oracle_clean` | クリーンラベルで学習（上限） |
| ref | `oracle_trusted{10,30}/irbl` | 無作為 10% / 30% を人手で直したと仮定して trusted にする（biquality-learn 本来の使い方） |

疑似 trusted の作り方（`split`）：

- `naive`：フラグなし全件を trusted、フラグ付きを untrusted
- `topconf`：フラグなしのうち、クラスごとに品質スコアの上位 30% を trusted、残り全部を untrusted
- `random`：フラグなしのうち、クラスごとに無作為な 30% を trusted、残り全部を untrusted

biquality-learn のアルゴリズム（`algo`）：IRBL、K-PDR、K-KMM（いずれも重み付け系）、GLC（LossCorrection）、Plugin、IRLNL（いずれも遷移行列系。行列は trusted から推定する `gold`）、trusted_only。

## 結果（全 90 条件の平均、baseline との差 pt）

詳細は `results/summary.md` と `results/delta_vs_baseline.png` を参照。

| 手法 | lr | hgb |
|---|---:|---:|
| oracle_clean（参考） | +7.95 | +11.95 |
| oracle_trusted30/irbl（参考・30% を人手で修正） | +6.64 | +8.56 |
| oracle_trusted10/irbl（参考・10% を人手で修正） | +4.48 | +5.64 |
| **bq/naive/irbl** | **+2.73** | **+3.56** |
| cl_prune | +2.28 | +3.25 |
| cl_cleanlearning | +1.53 | +3.19 |
| cl_weight/self_confidence | +2.32 | +1.99 |
| cl_feature | −5.86 | +3.46 |
| bq/naive/kpdr | +1.85 | +2.40 |
| bq/topconf/irbl | +0.70 | +2.41 |
| bq/random/glc | −4.37 | −11.84 |
| bq/naive/glc | −24.36 | −36.69 |
| bq/naive/plugin | −30.37 | −37.23 |

## 考察

1. **人手を使わない手法の中では `bq/naive/irbl` が最良だが、`cl_prune` との差は小さい。** 同じ条件どうしの差は lr で +0.45±0.36 pt（64% の条件で勝ち）、hgb で +0.31±0.24 pt（71% の条件で勝ち）。学習時間は約 3 倍かかる。cleanlab 標準の `CleanLearning`（クラス別の再重み付けあり）は `cl_prune` を上回らなかった（lr −0.75 pt、hgb −0.06 pt）。また `naive` の優位の一部は trusted が大きい（学習データの 53〜85%）ことによる。
2. **遷移行列系（GLC / Plugin / IRLNL）は疑似 trusted と組み合わせると大きく崩れる。** trusted のラベルが「ノイズ付きラベルそのもの」で、untrusted にノイズが偏って集まる（naive では untrusted の 39〜92% が誤り）ため、遷移行列が実態とかけ離れて推定される。ただし、無作為 10% を本物のクリーンラベルにした対照実験（`oracle_check`）でも遷移行列系は baseline を下回った（GLC: lr −2.62 / hgb −11.78 pt）。この設定ではそもそも弱い手法が、疑似 trusted でさらに大きく崩れた、というのが正確な解釈。`mechanism` の結果では、本物の trusted でも GLC は遷移行列の対角を過小推定する（untrusted の正ラベル率 0.70 に対し lr 0.50、hgb 0.62）。
3. **確信度上位で trusted を選ぶ `topconf` は、`naive` や `random` より悪い。** trusted が「簡単なサンプル」に偏るという選択バイアスの懸念が、数値として確認できた。特に covtype で顕著だった。
4. **スコアを重みにする方法は lr では cl_prune と同程度、hgb では劣る**（−1.26 pt）。
5. **スコアを特徴量にする方法はモデル次第。** hgb では cl_prune と同程度に効いたが、lr では大きく悪化した。線形モデルではスコアと他の特徴量の交互作用を表現できず、スコアはクラスごとの切片をずらす働きしかしないため。
6. **instance ノイズ（現実の誤アノテーションに近い）では、どの自動手法も伸びが小さい。** cleanlab の再現率は 37〜74% にとどまる。pair ノイズ 40% では、10% を人手で直すだけで +20 pt 前後改善するのに対し、自動手法は +7 pt 程度で、差が大きい。
7. **trusted の大きさをそろえると人手確認の優位ははっきりする。** 30% を人手で直した IRBL は `bq/naive/irbl` を lr +3.91 pt（勝率 72%）、hgb +5.00 pt（勝率 92%）上回った。10% では平均では上回るが勝率は半数未満。
