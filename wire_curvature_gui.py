#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ワイヤ曲率計測 GUI  (Wire Curvature Analyzer)
-------------------------------------------------
青背景で撮影したワイヤ曲げ実験動画から、伸びているワイヤの
中心線に沿った局所曲率 kappa(s) = 1/R を画像処理で求める
デスクトップアプリ。

処理フロー:
  1. グレースケール閾値でワイヤ(明るい銀色)を背景(青)から分離
  2. 太い塊(機構ヘッド・センサ等)を形態学的に除去し、細い線構造を抽出
  3. 最長連結成分を選び skeletonize で1px中心線化
  4. 中心線を根元→先端に順序付け、平滑化スプラインを当てはめ
  5. kappa(s) = |x'y'' - y'x''| / (x'^2+y'^2)^{3/2} を弧長に沿って計算
  6. 2点クリック校正(既知寸法入力)で px -> mm 換算し 1/mm で出力

操作(直感優先の刷新版):
  - 「動画を開く」で mp4 を読み込み、スライダーでフレーム移動
  - 「画像を開く」で静止画(png/jpg等)を1枚読み込み、そのまま解析・校正・CSV書出し
  - ROI: 画像上でドラッグして解析範囲を指定
  - ★ライブプレビュー: スライダーを動かすと抽出マスクが赤の半透明で即時表示
  - ★「明るさ自動(Otsu)」: ROI内の輝度分布から閾値を自動設定
  - ★「ワイヤをクリックで調整」: ワイヤ上を1回クリックして閾値を自動推定
  - 「解析」で中心線曲率を計算、失敗時は原因別のヒントを表示
  - 「校正(2点)」→ 既知長さの2点クリック→ mm 入力で px→mm 換算
    (クリック中はカーソル周辺を隅のルーペで拡大表示し、端点を精密に合わせられる)
    (スケール px/mm が既知なら「px/mm 直接入力」欄に直打ち→適用でもよい)
  - CSV / 注釈画像 / 全フレーム一括 で書き出し

依存: opencv-python, numpy, scipy, scikit-image, matplotlib  (Tkinterは標準)
実行: python wire_curvature_gui.py
"""
import os, sys, collections
import numpy as np
import cv2

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import font_manager, rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector


def _setup_japanese_font():
    """matplotlibの図中日本語が豆腐(□)化しないよう、環境にある日本語フォントを設定。"""
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Noto Sans JP", "IPAexGothic", "TakaoGothic",
                  "Hiragino Sans", "Hiragino Maru Gothic Pro"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False   # マイナス記号の豆腐も防ぐ


_setup_japanese_font()

from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev


# ----------------------------------------------------------------------
# 画像処理コア
# ----------------------------------------------------------------------
def wire_stat(img, use_color=True):
    """閾値化に使う単一チャネル統計量を返す。
    use_color=True: 各画素の最小チャネル(min(B,G,R))。銀/白ワイヤは全チャネルが
      高いので大きく、彩度の高い青背景は R が低いので小さい。→ 明暗ムラに強く、
      暗いワイヤ部分も背景と分離できる(青背景×銀ワイヤに最適)。
    use_color=False: 従来どおりのグレースケール輝度。"""
    if use_color:
        return img.min(axis=2).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def wire_mask(img, roi, thr=150, blob=17, close=11, minarea=300,
              use_color=True):
    """ワイヤ二値マスクを作る。roi=(x0,y0,x1,y1)
    use_color で色分離(最小チャネル)/従来のグレースケールを切替。"""
    x0, y0, x1, y1 = roi
    stat = wire_stat(img, use_color)
    _, bw = cv2.threshold(stat, thr, 255, cv2.THRESH_BINARY)
    m = np.zeros(img.shape[:2], np.uint8)
    m[y0:y1, x0:x1] = 255
    bw = cv2.bitwise_and(bw, m)
    # 太い塊(機構ヘッド/センサ)を除去 -> 細い線構造のみ残す
    if blob and blob >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blob, blob))
        thick = cv2.dilate(cv2.morphologyEx(bw, cv2.MORPH_OPEN, k),
                           np.ones((5, 5), np.uint8))
        bw = cv2.bitwise_and(bw, cv2.bitwise_not(thick))
    # 途切れをつなぐ: 膨張で断片を橋渡しする。細線は後段の skeletonize で
    # 中心線化するので、太らせても中心線は保たれる。閉(close)より大きなギャップを
    # 繋げられるので、断片化した細いワイヤの再結合に効く。
    if close and close >= 3:
        bw = cv2.dilate(
            bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    # 最長(対角最大)の連結成分をワイヤとして採用
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    best, bi = 0, -1
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < minarea:
            continue
        d = (w * w + h * h) ** 0.5
        if d > best:
            best, bi = d, i
    if bi < 0:
        return (bw > 0).astype(np.uint8)
    return (lab == bi).astype(np.uint8)


def _build_neighbors(pts):
    """スケルトン画素の8近傍隣接リストを作る。"""
    idx = {tuple(p): i for i, p in enumerate(pts)}
    off = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1)]
    nb = [[] for _ in pts]
    for i, (r, c) in enumerate(pts):
        for dr, dc in off:
            j = idx.get((r + dr, c + dc))
            if j is not None:
                nb[i].append(j)
    return nb


def _walk_cycle(nb, start):
    """端点の無い(閉じた)スケルトンを、隣接に沿って一周たどって順序付ける。
    次数2を仮定した貪欲ウォーク。分岐で乱れたら None を返し呼び出し側で退避。"""
    n = len(nb)
    order = [start]
    visited = {start}
    prev, cur = -1, start
    while True:
        cand = [v for v in nb[cur] if v != prev]
        unv = [v for v in cand if v not in visited]
        if unv:
            nxt = unv[0]
        elif start in nb[cur] and len(order) > 3:
            break                     # 一周して戻れた(閉ループ完成)
        else:
            break                     # 行き止まり/分岐で追跡不能
        prev, cur = cur, nxt
        order.append(cur)
        visited.add(cur)
        if len(order) > n:
            break
    # 過半数の画素をたどれていれば閉ループとして採用
    if len(order) >= max(12, int(0.6 * n)):
        return np.array(order)
    return None


def _order_skeleton(sk):
    """スケルトン画素を中心線に沿って順序付ける。
    返り値: (順序付き(row,col)配列, is_closed) / 失敗時 (None, False)
    - 端点が2つ以上ある通常の弧: 端点間の最長パス(BFS×2)
    - 端点が無い閉ループ: サイクルを一周たどる(周期スプライン用)"""
    pts = np.column_stack(np.where(sk > 0))
    if len(pts) < 5:
        return None, False
    nb = _build_neighbors(pts)

    def bfs(s):
        dist = {s: 0}
        par = {s: -1}
        q = collections.deque([s])
        far = s
        while q:
            u = q.popleft()
            for v in nb[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                    if dist[v] > dist[far]:
                        far = v
        return far, dist, par

    def longest_path():
        a, _, _ = bfs(0)
        b, _, par = bfs(a)            # a->b が最長パス
        path = []
        u = b
        while u != -1:
            path.append(u)
            u = par[u]
        return pts[path[::-1]]

    endpoints = [i for i in range(len(pts)) if len(nb[i]) == 1]
    if len(endpoints) == 0:
        # 端点なし = 閉ループ。まず一周たどる。失敗したら最長パスに退避。
        ordered = _walk_cycle(nb, 0)
        if ordered is not None:
            return pts[ordered], True
        return longest_path(), False
    return longest_path(), False


def curvature_of(mask, smooth_scale=2.0, npts=300):
    """マスク -> 中心線曲率。返り値 dict(x,y,kappa,s,total_len) 単位は px / 1/px"""
    sk = skeletonize(mask > 0)
    ordered, closed = _order_skeleton(sk.astype(np.uint8))
    if ordered is None or len(ordered) < 12:
        return None
    y = ordered[:, 0].astype(float)
    x = ordered[:, 1].astype(float)
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if d[-1] <= 0:
        return None
    s_par = d / d[-1]
    smooth = max(len(x) * float(smooth_scale), 1.0)
    per = 1 if closed else 0
    try:
        tck, _ = splprep([x, y], u=s_par, s=smooth, k=3, per=per)
    except Exception:
        if per:                        # 周期スプラインが不調なら開曲線で再試行
            try:
                tck, _ = splprep([x, y], u=s_par, s=smooth, k=3, per=0)
            except Exception:
                return None
        else:
            return None
    uu = np.linspace(0, 1, npts)
    xs, ys = splev(uu, tck)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom[denom == 0] = np.nan
    # 符号付き曲率(絶対値を取らない)。符号は曲がる向きを表す。
    # 表示側で絶対値/符号付きを切替できる(App._kappa_signed)。
    kappa = (dx * ddy - dy * ddx) / denom            # 1/px (signed)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]  # px
    return dict(x=xs, y=ys, kappa=kappa, s=s, total_len=s[-1],
                closed=closed)


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("ワイヤ曲率計測 GUI")
        self.cap = None
        self.nframes = 0
        self.fps = 30.0
        self.frame = None            # 現在のBGRフレーム
        self.result = None           # curvature_of の結果
        self.roi = None              # (x0,y0,x1,y1)
        self.px_per_mm = None        # 校正スケール
        self.calib_mode = False
        self.calib_pts = []
        self.pick_mode = False       # ワイヤクリックで閾値調整
        self.video_path = None
        self.cur_index = 0

        # ライブプレビュー用の状態
        self._preview_job = None     # after() のハンドル(デバウンス)
        self._img_artist = None      # 背景画像 AxesImage
        self._overlay_artist = None  # マスク重ね描き AxesImage
        self._slider_labels = {}     # スライダー名 -> 値ラベル

        # フレーム移動のデバウンス(高速スクロール時の描画詰まり防止)
        self._frame_job = None
        self._pending_frame = 0

        # 校正ルーペ(2点クリック時にカーソル周辺を拡大表示する拡大鏡)
        self._loupe_ax = None       # 拡大表示用インセット軸
        self._loupe_im = None       # 拡大画像 AxesImage
        self._loupe_vline = None     # 中心十字(縦)
        self._loupe_hline = None     # 中心十字(横)
        self._loupe_txt = None       # 座標ラベル
        self._loupe_bg = None        # ブリット用の背景キャッシュ
        self._loupe_half = 12        # 取り込む半径[px](実サイズ 2*half+1 角)
        self._loupe_left = None      # 現在の表示側(True=左, False=右)

        # 解析範囲トリム(下グラフの縦棒で端のノイズを除外する)
        self.s_lo_frac = 0.0     # 左棒: 弧長に対する割合 [0,1]
        self.s_hi_frac = 1.0     # 右棒
        self._trim_lines = []    # [左Line2D, 右Line2D]
        self._drag_idx = None    # ドラッグ中の棒 (0=左,1=右) / None

        self._build_controls()
        self._build_figures()

        # 起動時にサンプル動画があれば自動読み込み
        for cand in ("experiment_k20.mp4",
                     os.path.join(os.path.dirname(__file__), "experiment_k20.mp4")):
            if os.path.exists(cand):
                self.open_video(cand)
                break

    # -- 左: コントロールパネル --------------------------------------
    def _build_controls(self):
        p = ttk.Frame(self.root, padding=8)
        p.grid(row=0, column=0, sticky="ns")
        r = 0

        of = ttk.Frame(p)
        of.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 6)); r += 1
        ttk.Button(of, text="動画を開く", command=self.on_open).pack(
            side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(of, text="画像を開く", command=self.on_open_image).pack(
            side="left", expand=True, fill="x", padx=(2, 0))

        self.lbl_file = ttk.Label(p, text="(未読み込み)", width=32, foreground="#555")
        self.lbl_file.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        # フレームスライダ
        ttk.Label(p, text="フレーム").grid(row=r, column=0, sticky="w", pady=(8, 0)); r += 1
        self.v_frame = tk.IntVar(value=0)
        self.s_frame = ttk.Scale(p, from_=0, to=0, orient="horizontal",
                                 command=self._on_frame_slider)
        self.s_frame.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        for evt in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.s_frame.bind(evt, self._on_frame_wheel)
        self.lbl_frame = ttk.Label(p, text="0 / 0")
        self.lbl_frame.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        # ライブプレビュー トグル
        sep0 = ttk.Separator(p); sep0.grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        self.v_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="ライブプレビュー(抽出マスクを赤で重ねる)",
                        variable=self.v_preview,
                        command=self._on_preview_toggle).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        # 青背景を色で分離(細い/暗いワイヤの断片化に有効)
        self.v_color = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            p, text="青背景を色で分離(推奨: 細い銀ワイヤ)",
            variable=self.v_color,
            command=self._on_color_toggle).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        # 各種スライダ(分かりやすいラベル + 一行ヘルプ)
        self.v_thr = self._slider(
            p, "thr", "① 明るさしきい値", 0, 255, 150,
            help="これより明るい画素をワイヤとみなす。ワイヤが消えるなら下げる"
                 "(色分離ON時は各画素の最小チャネルに対する閾値)")
        # 閾値の補助ボタン(自動 / クリック調整)
        rr = self._slider_next_row
        bf = ttk.Frame(p); bf.grid(row=rr, column=0, columnspan=2, sticky="ew"); r = rr + 1
        ttk.Button(bf, text="明るさ自動(Otsu)", command=self.auto_threshold).pack(
            side="left", expand=True, fill="x", padx=(0, 2))
        self.btn_pick = ttk.Button(bf, text="ワイヤをクリックで調整", command=self.start_pick)
        self.btn_pick.pack(side="left", expand=True, fill="x", padx=(2, 0))
        self._slider_next_row = r

        self.v_blob = self._slider(
            p, "blob", "② 太い部分を消す", 0, 41, 17,
            help="ヘッドやセンサの太い塊を除去。ワイヤまで消えるなら下げる")
        self.v_close = self._slider(
            p, "close", "③ 線をつなぐ", 0, 41, 11,
            help="途切れた線を橋渡し。断片化して抽出できないなら上げる")
        self.v_smooth = self._slider(
            p, "smooth", "④ なめらかさ", 100, 8000, 2000,
            help="曲率のノイズ抑制。ガタつくなら上げ、鈍るなら下げる",
            scale=0.001, step=50)
        r = self._slider_next_row

        ttk.Button(p, text="解析 (このフレーム)", command=self.analyze).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(8, 2)); r += 1

        # グラフの縦軸: 曲率κ / 曲率半径R の切替
        gf = ttk.Frame(p); gf.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(gf, text="グラフ:").pack(side="left")
        self.v_radius = tk.BooleanVar(value=False)
        ttk.Radiobutton(gf, text="曲率 κ", variable=self.v_radius, value=False,
                        command=self._on_metric_toggle).pack(side="left")
        ttk.Radiobutton(gf, text="曲率半径 R", variable=self.v_radius, value=True,
                        command=self._on_metric_toggle).pack(side="left")
        # 符号付き(±): 負の曲率(曲がる向き)も表示する
        self.v_signed = tk.BooleanVar(value=True)
        ttk.Checkbutton(gf, text="符号付き(±)", variable=self.v_signed,
                        command=self._on_metric_toggle).pack(side="left",
                                                             padx=(8, 0))
        # 図のレイアウト: 縦並び / 横並び(縦長画像で図が大きく見やすくなる)
        lf = ttk.Frame(p); lf.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(lf, text="レイアウト:").pack(side="left")
        self.v_layout = tk.StringVar(value="stacked")
        ttk.Radiobutton(lf, text="縦並び", variable=self.v_layout,
                        value="stacked",
                        command=self._on_layout_change).pack(side="left")
        ttk.Radiobutton(lf, text="横並び(縦長向き)", variable=self.v_layout,
                        value="side",
                        command=self._on_layout_change).pack(side="left")

        # ROI
        rf = ttk.Frame(p); rf.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        ttk.Label(rf, text="ROI: 画像上をドラッグ", foreground="#555").pack(side="left")
        ttk.Button(rf, text="全体にリセット", command=self.reset_roi).pack(side="right")

        # 校正
        sep = ttk.Separator(p); sep.grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        self.btn_calib = ttk.Button(p, text="校正 (2点クリック)", command=self.start_calib)
        self.btn_calib.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self.lbl_scale = ttk.Label(p, text="スケール: 未校正 (px単位で表示)", foreground="#a00")
        self.lbl_scale.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 0)); r += 1
        # スケール直接入力(2点クリックの代わりに px/mm を直打ち)
        me = ttk.Frame(p); me.grid(row=r, column=0, columnspan=2, sticky="ew",
                                   pady=(2, 0)); r += 1
        ttk.Label(me, text="px/mm 直接入力:").pack(side="left")
        self.v_scale = tk.StringVar()
        e_scale = ttk.Entry(me, textvariable=self.v_scale, width=9)
        e_scale.pack(side="left", padx=4)
        e_scale.bind("<Return>", lambda ev: self.apply_manual_scale())
        ttk.Button(me, text="適用", command=self.apply_manual_scale).pack(
            side="left")
        ttk.Button(p, text="校正クリア", command=self.clear_calib).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(2, 0)); r += 1

        # 出力
        sep2 = ttk.Separator(p); sep2.grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Button(p, text="このフレームCSV書出し", command=self.export_frame_csv).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=1); r += 1
        ttk.Button(p, text="注釈画像を保存", command=self.save_annotated).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=1); r += 1
        ttk.Button(p, text="全フレーム一括解析", command=self.batch_all).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=1); r += 1

        # 結果 / ヒント表示
        sep3 = ttk.Separator(p); sep3.grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        self.txt = tk.Text(p, width=36, height=9, font=("TkFixedFont", 9))
        self.txt.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self._log("動画を開いてROIをドラッグ→スライダーを動かすと、\n"
                  "赤いマスクがワイヤに重なるよう調整できます。\n"
                  "きれいに乗ったら『解析』を押してください。")

        p.columnconfigure(0, weight=1)

    def _bind_wheel(self, scale, lo, hi, step=1):
        """ttk.Scale をマウスホイールで増減できるようにする。
        set() が command を呼ぶので、ラベル更新とライブプレビューも自動で走る。"""
        def handler(ev):
            cur = int(float(scale.get()))
            down = getattr(ev, "num", None) == 5 or getattr(ev, "delta", 0) < 0
            newv = int(min(hi, max(lo, cur + (-step if down else step))))
            if newv != cur:
                scale.set(newv)
            return "break"          # 図のスクロール等に伝播させない
        scale.bind("<MouseWheel>", handler)   # Windows / macOS
        scale.bind("<Button-4>", handler)     # Linux 上
        scale.bind("<Button-5>", handler)     # Linux 下

    def _slider(self, parent, name, label, lo, hi, init, help="", scale=1.0,
                step=1):
        """ラベル+値+一行ヘルプ付きスライダ。値変更でライブプレビューを予約。
        scale: 表示/内部値の倍率(smoothは0.001)。get()時は生値、表示は生値のまま。
        step: マウスホイール1ノッチあたりの増減量。"""
        row = getattr(self, "_slider_next_row", None)
        if row is None:
            # 既存gridの続きに積む: フレーム内で行番号を自前管理
            row = self._slider_start_row()
        head = ttk.Frame(parent)
        head.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(head, text=label).pack(side="left")
        val = ttk.Label(head, text=str(init), width=6, anchor="e")
        val.pack(side="right")
        self._slider_labels[name] = val

        var = tk.IntVar(value=init)

        def cb(_):
            val.config(text=str(int(float(sc.get()))))
            self._schedule_preview()
        sc = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal",
                       variable=var, command=cb)
        sc.grid(row=row + 1, column=0, columnspan=2, sticky="ew")
        self._bind_wheel(sc, lo, hi, step)
        if help:
            ttk.Label(parent, text=help, foreground="#777",
                      wraplength=250, font=("TkDefaultFont", 8)).grid(
                row=row + 2, column=0, columnspan=2, sticky="w")
            self._slider_next_row = row + 3
        else:
            self._slider_next_row = row + 2
        return var

    def _slider_start_row(self):
        # 最初のスライダ開始行(フレーム/プレビュー領域の直後あたり)
        return 8

    def _set_slider(self, name, var, value):
        """プログラムからスライダ値を設定し、ラベルとプレビューも更新。"""
        value = int(value)
        var.set(value)
        if name in self._slider_labels:
            self._slider_labels[name].config(text=str(value))
        self._schedule_preview()

    # -- 右: 図 ------------------------------------------------------
    def _build_figures(self):
        f = ttk.Frame(self.root)
        f.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(12, 8))
        self.cbar = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # イベント接続は一度きり(軸を張り替えてもハンドラは self.ax_* を都度参照)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_calib_motion)
        self.canvas.mpl_connect("button_press_event", self._on_trim_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_trim_motion)
        self.canvas.mpl_connect("button_release_event", self._on_trim_release)
        # 実際の軸配置(縦並び/横並び)を構築
        self._layout_axes()

    def _layout_axes(self):
        """現在の self.v_layout に従って軸を(再)構築する。
        縦並び(stacked): 上=画像+カラーバー, 下=曲率グラフ。
        横並び(side): 左=画像+カラーバー, 右=曲率グラフ(縦長画像で無駄な余白を解消)。
        カラーバーは専用の固定枠(cax)に描き、操作のたびに画像が縮むのを防ぐ。"""
        self.fig.clf()
        if getattr(self, "v_layout", None) is not None and \
                self.v_layout.get() == "side":
            # 画像 / カラーバー / スペーサ(y軸ラベル用) / グラフ
            gs = self.fig.add_gridspec(
                1, 4, width_ratios=[0.5, 0.03, 0.11, 1.0], wspace=0.04)
            self.ax_img = self.fig.add_subplot(gs[0, 0])
            self.cax = self.fig.add_subplot(gs[0, 1])
            self.ax_k = self.fig.add_subplot(gs[0, 3])
            self.fig.subplots_adjust(left=0.045, right=0.965, top=0.93,
                                     bottom=0.10)
        else:
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 0.035],
                                       height_ratios=[1, 1],
                                       wspace=0.03, hspace=0.28)
            self.ax_img = self.fig.add_subplot(gs[0, 0])
            self.cax = self.fig.add_subplot(gs[0, 1])
            self.ax_k = self.fig.add_subplot(gs[1, :])
            self.fig.subplots_adjust(left=0.07, right=0.92, top=0.94,
                                     bottom=0.08)
        self.cax.set_visible(False)
        self.ax_img.set_title("画像 + 抽出マスク(赤)  — ドラッグでROI指定")
        self.ax_k.set_title("局所曲率分布")
        self.cbar = None
        # clf で消えた重ね描き/ルーペ関連の参照をリセット
        self._img_artist = None
        self._overlay_artist = None
        self._loupe_ax = None
        self._loupe_bg = None
        # ROI 選択を新しい軸に張り直す
        self.selector = RectangleSelector(
            self.ax_img, self._on_roi, useblit=True, button=[1],
            minspanx=10, minspany=10, spancoords="pixels", interactive=True)
        # 現在の内容を再描画
        if getattr(self, "result", None) is not None:
            self._plot()
        elif getattr(self, "frame", None) is not None:
            self.show_frame()
            self.update_preview()
        else:
            self.canvas.draw_idle()

    def _on_layout_change(self):
        """レイアウト(縦並び/横並び)切替。軸を張り替えて再描画。"""
        self._layout_axes()

    def _auto_layout(self):
        """読み込み時、ROI の縦横比でレイアウトを自動選択(縦長→横並び)。
        手動切替を上書きしないよう、値が変わるときだけ張り替える。"""
        if self.roi is None:
            return
        x0, y0, x1, y1 = self.roi
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        want = "side" if h > 1.8 * w else "stacked"
        if self.v_layout.get() != want:
            self.v_layout.set(want)
            self._layout_axes()

    # -- 動画 --------------------------------------------------------
    def on_open(self):
        path = filedialog.askopenfilename(
            title="動画ファイルを選択",
            filetypes=[("動画", "*.mp4 *.avi *.mov *.mkv"), ("すべて", "*.*")])
        if path:
            self.open_video(path)

    def on_open_image(self):
        path = filedialog.askopenfilename(
            title="画像ファイルを選択",
            filetypes=[("画像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("すべて", "*.*")])
        if path:
            self.open_image(path)

    @staticmethod
    def _imread_unicode(path):
        """日本語等の非ASCIIパスでも読めるよう imdecode 経由で画像を読む。"""
        try:
            data = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def open_image(self, path):
        img = self._imread_unicode(path)
        if img is None:
            messagebox.showerror("エラー", "画像を開けませんでした")
            return
        # 動画を開いていたら解放し、静止画1枚を単一フレームとして扱う
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.video_path = path
        self.frame = img
        self.nframes = 1
        self.cur_index = 0
        self.fps = 1.0
        self.result = None
        self.s_lo_frac, self.s_hi_frac = 0.0, 1.0   # トリムを全域にリセット
        self.lbl_file.config(text=os.path.basename(path))
        self.s_frame.config(to=0)          # 静止画なのでスライダは無効(1枚)
        self.s_frame.set(0)
        self.lbl_frame.config(text="静止画 (1枚)")
        H, W = self.frame.shape[:2]
        # 既定ROI(画面右側の自由弧領域)
        self.roi = (int(0.54 * W), int(0.07 * H), int(0.79 * W), int(0.83 * H))
        self.show_frame()
        self._draw_roi_rect()
        # ROI内の輝度から閾値を自動初期化
        self.auto_threshold(quiet=True)
        self.update_preview()
        self._auto_layout()            # 縦長ならレイアウトを横並びに自動切替
        self._log(f"画像読み込み: {os.path.basename(path)}\n"
                  f"{W}x{H}\n"
                  f"閾値を自動設定しました。赤マスクを見ながら微調整を。")

    def open_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("エラー", "動画を開けませんでした")
            return
        self.cap = cap
        self.video_path = path
        self.s_lo_frac, self.s_hi_frac = 0.0, 1.0   # トリムを全域にリセット
        self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.lbl_file.config(text=os.path.basename(path))
        self.s_frame.config(to=max(self.nframes - 1, 0))
        self.load_frame(0)
        # 既定ROI(画面右側の自由弧領域)
        H, W = self.frame.shape[:2]
        self.roi = (int(0.54 * W), int(0.07 * H), int(0.79 * W), int(0.83 * H))
        self.show_frame()
        self._draw_roi_rect()
        # ROI内の輝度から閾値を自動初期化(勘値からの脱却)
        self.auto_threshold(quiet=True)
        self.update_preview()
        self._auto_layout()            # 縦長ならレイアウトを横並びに自動切替
        self._log(f"読み込み: {os.path.basename(path)}\n"
                  f"{W}x{H}, {self.nframes}フレーム, {self.fps:.1f}fps\n"
                  f"閾値を自動設定しました。赤マスクを見ながら微調整を。")

    def load_frame(self, i):
        i = int(max(0, min(i, self.nframes - 1)))
        # 1コマ前進(=次フレーム)なら、圧縮動画で重いシークを省いて逐次読みする
        if i == self.cur_index + 1:
            ok, fr = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = self.cap.read()
        if ok:
            self.frame = fr
            self.cur_index = i
            self.lbl_frame.config(text=f"{i} / {self.nframes-1}  "
                                       f"({i/self.fps:.2f}s)")

    def _on_frame_wheel(self, ev):
        """フレームスライダをホイールで1フレームずつ送る(上限は動画長)。"""
        if self.cap is None:
            return "break"
        cur = int(float(self.s_frame.get()))
        down = getattr(ev, "num", None) == 5 or getattr(ev, "delta", 0) < 0
        newv = max(0, min(self.nframes - 1, cur + (-1 if down else 1)))
        if newv != cur:
            self.s_frame.set(newv)   # command(_on_frame_slider) が発火
        return "break"

    def _on_frame_slider(self, _):
        if self.cap is None:
            return
        i = int(float(self.s_frame.get()))
        self._pending_frame = i
        # ラベルだけ即時更新(軽い)。重いデコード/再描画はデバウンスして最後だけ実行
        self.lbl_frame.config(text=f"{i} / {self.nframes-1}  ({i/self.fps:.2f}s)")
        if self._frame_job is not None:
            self.root.after_cancel(self._frame_job)
        self._frame_job = self.root.after(70, self._render_pending_frame)

    def _render_pending_frame(self):
        self._frame_job = None
        self.load_frame(self._pending_frame)
        self.result = None
        self.show_frame()
        self.update_preview()

    # -- ROI / 校正 / クリック調整 -----------------------------------
    def _on_roi(self, e1, e2):
        x0, x1 = sorted((int(e1.xdata), int(e2.xdata)))
        y0, y1 = sorted((int(e1.ydata), int(e2.ydata)))
        self.roi = (x0, y0, x1, y1)
        self._log(f"ROI設定: x[{x0}-{x1}] y[{y0}-{y1}]")
        self.update_preview()

    def reset_roi(self):
        if self.frame is None:
            return
        H, W = self.frame.shape[:2]
        self.roi = (0, 0, W, H)
        self._draw_roi_rect()
        self._log("ROIを画像全体にリセットしました")
        self.update_preview()

    def _draw_roi_rect(self):
        if self.roi:
            self.selector.extents = (self.roi[0], self.roi[2],
                                     self.roi[1], self.roi[3])

    def start_calib(self):
        self.calib_mode = True
        self.pick_mode = False
        self.calib_pts = []
        self.selector.set_active(False)
        self.btn_calib.config(text="…2点をクリック")
        self._ensure_loupe()
        self._loupe_bg = None        # ブリット背景を取り直す
        self._log("校正: 既知長さの端点2つを画像上でクリック\n"
                  "(カーソル周辺を右上/左上のルーペで拡大表示します)")

    def clear_calib(self):
        self.px_per_mm = None
        self.calib_mode = False
        self.calib_pts = []
        self.btn_calib.config(text="校正 (2点クリック)")
        self._hide_loupe()
        self.v_scale.set("")
        self.lbl_scale.config(text="スケール: 未校正 (px単位で表示)", foreground="#a00")
        self._log("校正をクリアしました")
        if self.result:
            self._plot()

    def apply_manual_scale(self):
        """入力欄の px/mm 値をスケールとして直接適用(2点クリックの代替)。"""
        txt = self.v_scale.get().strip()
        if not txt:
            return
        try:
            val = float(txt)
        except ValueError:
            messagebox.showerror("エラー", "数値を入力してください (px/mm)")
            return
        if val <= 0:
            messagebox.showerror("エラー", "0より大きい値を入力してください")
            return
        self.px_per_mm = val
        self.v_scale.set(f"{val:.4g}")
        self.lbl_scale.config(text=f"スケール: {val:.3f} px/mm", foreground="#080")
        self._log(f"スケールを手動設定: {val:.3f} px/mm")
        if self.result:
            self._plot()

    def start_pick(self):
        """ワイヤ上を1回クリック→その明るさから閾値を推定するモード。"""
        if self.frame is None:
            return
        self.pick_mode = True
        self.calib_mode = False
        self._hide_loupe()
        self.btn_calib.config(text="校正 (2点クリック)")
        self.selector.set_active(False)
        self.btn_pick.config(text="…ワイヤをクリック")
        self._log("ワイヤの明るい部分を1回クリックしてください。\n"
                  "その輝度を基準に閾値を自動設定します。")

    def _end_pick(self):
        self.pick_mode = False
        self.btn_pick.config(text="ワイヤをクリックで調整")
        self.selector.set_active(True)

    def _on_click(self, ev):
        if ev.inaxes != self.ax_img or ev.xdata is None:
            return
        # --- ワイヤクリックで閾値調整 ---
        if self.pick_mode:
            cx, cy = int(ev.xdata), int(ev.ydata)
            gray = wire_stat(self.frame, self.v_color.get())
            H, W = gray.shape
            x0, x1 = max(0, cx - 4), min(W, cx + 5)
            y0, y1 = max(0, cy - 4), min(H, cy + 5)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                self._end_pick()
                return
            v = float(np.median(patch))
            # クリックした明るさより少し下を閾値に(背景との中間を狙う)
            thr = int(np.clip(v - 45, 5, 254))
            self._set_slider("thr", self.v_thr, thr)
            self._end_pick()
            self._log(f"クリック輝度≈{v:.0f} → 閾値を {thr} に設定。\n"
                      f"赤マスクを見て必要ならスライダーで微調整を。")
            return
        # --- 校正クリック ---
        if not self.calib_mode:
            return
        self.calib_pts.append((ev.xdata, ev.ydata))
        self.ax_img.plot(ev.xdata, ev.ydata, "y+", ms=12, mew=2)
        self.canvas.draw_idle()
        self._loupe_bg = None       # 1点目のマーカーを含めて背景を取り直す
        if len(self.calib_pts) == 2:
            self._hide_loupe()
            (x0, y0), (x1, y1) = self.calib_pts
            dpx = float(np.hypot(x1 - x0, y1 - y0))
            mm = simpledialog.askfloat(
                "校正", f"2点間の実寸を入力 [mm]\n(画面上 {dpx:.1f} px)",
                minvalue=0.001)
            self.calib_mode = False
            self.btn_calib.config(text="校正 (2点クリック)")
            self.selector.set_active(True)
            if mm:
                self.px_per_mm = dpx / mm
                self.v_scale.set(f"{self.px_per_mm:.4g}")
                self.lbl_scale.config(
                    text=f"スケール: {self.px_per_mm:.3f} px/mm", foreground="#080")
                self._log(f"校正完了: {dpx:.1f}px = {mm}mm "
                          f"→ {self.px_per_mm:.3f} px/mm")
                if self.result:
                    self._plot()
            else:
                self.show_frame()
                self.update_preview()

    # -- 校正ルーペ(カーソル周辺の拡大鏡) ---------------------------
    def _ensure_loupe(self):
        """拡大鏡インセットを一度だけ生成(画像軸の上に重ねる浮遊軸)。"""
        if self._loupe_ax is not None:
            return
        ax = self.fig.add_axes([0.70, 0.60, 0.20, 0.20], zorder=20)
        ax.set_aspect("equal")           # 拡大画素を正方形に保つ
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color("#ff0"); sp.set_linewidth(1.5)
        self._loupe_im = ax.imshow(np.zeros((1, 1, 3), np.uint8),
                                   interpolation="nearest", zorder=1)
        self._loupe_vline = ax.axvline(0, color="#0f0", lw=0.8, alpha=0.9)
        self._loupe_hline = ax.axhline(0, color="#0f0", lw=0.8, alpha=0.9)
        self._loupe_txt = ax.text(
            0.03, 0.97, "", transform=ax.transAxes, va="top", ha="left",
            color="#ff0", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="black",
                      ec="none", alpha=0.5))
        ax.set_visible(False)
        self._loupe_ax = ax

    def _place_loupe(self, left):
        """画像軸の上端に沿って、カーソルと反対側の隅にルーペを配置。"""
        pos = self.ax_img.get_position()
        w = 0.30 * pos.width
        h = 0.42 * pos.height
        m = 0.008
        y = pos.y0 + pos.height - h - m
        x = pos.x0 + m if left else pos.x0 + pos.width - w - m
        self._loupe_ax.set_position([x, y, w, h])

    def _loupe_patch(self, cx, cy, half):
        """(cx,cy)中心・半径halfのRGBパッチと画像座標の範囲を返す(端は黒詰め)。"""
        H, W = self.frame.shape[:2]
        x0, x1 = cx - half, cx + half + 1
        y0, y1 = cy - half, cy + half + 1
        bgr = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            bgr[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                self.frame[sy0:sy1, sx0:sx1]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (x0, x1, y0, y1)

    def _on_calib_motion(self, ev):
        """校正中、カーソルが画像上にある間だけルーペを追従表示。"""
        if not self.calib_mode or self.frame is None:
            return
        if ev.inaxes != self.ax_img or ev.xdata is None:
            return
        self._show_loupe(int(round(ev.xdata)), int(round(ev.ydata)))

    def _show_loupe(self, cx, cy):
        self._ensure_loupe()
        half = self._loupe_half
        patch, (x0, x1, y0, y1) = self._loupe_patch(cx, cy, half)
        # カーソルが右半分ならルーペは左上へ(注視点を隠さない)
        left = cx > self.frame.shape[1] * 0.5
        if left != self._loupe_left or self._loupe_bg is None:
            self._loupe_left = left
            self._place_loupe(left)
        ax = self._loupe_ax
        self._loupe_im.set_data(patch)
        self._loupe_im.set_extent((x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        ax.set_xlim(x0 - 0.5, x1 - 0.5)
        ax.set_ylim(y1 - 0.5, y0 - 0.5)      # 画像座標なので下向きに反転
        self._loupe_vline.set_xdata([cx, cx])
        self._loupe_hline.set_ydata([cy, cy])
        self._loupe_txt.set_text(f"({cx}, {cy})")
        # 背景を一度だけ取り込み(ルーペを隠した状態で撮る)→以後はブリット
        if self._loupe_bg is None:
            ax.set_visible(False)
            self.canvas.draw()
            self._loupe_bg = self.canvas.copy_from_bbox(self.fig.bbox)
        ax.set_visible(True)
        self.canvas.restore_region(self._loupe_bg)
        ax.draw(self.canvas.get_renderer())
        self.canvas.blit(self.fig.bbox)

    def _hide_loupe(self):
        if self._loupe_ax is not None and self._loupe_ax.get_visible():
            self._loupe_ax.set_visible(False)
            self._loupe_bg = None
            self.canvas.draw_idle()
        else:
            self._loupe_bg = None

    # -- 閾値自動 ----------------------------------------------------
    def _on_color_toggle(self):
        """色分離ON/OFF切替。統計量が変わるので閾値を取り直してプレビュー更新。"""
        if self.frame is None:
            return
        self.auto_threshold(quiet=True)
        self._schedule_preview()

    def auto_threshold(self, quiet=False):
        """ROI内(なければ全体)の統計量からOtsu閾値を求めてスライダに反映。
        色分離ONなら最小チャネル、OFFならグレースケールに対して計算する。"""
        if self.frame is None:
            return
        gray = wire_stat(self.frame, self.v_color.get())
        if self.roi:
            x0, y0, x1, y1 = self.roi
            sub = gray[y0:y1, x0:x1]
        else:
            sub = gray
        if sub.size == 0:
            return
        t, _ = cv2.threshold(sub, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._set_slider("thr", self.v_thr, int(t))
        if not quiet:
            self._log(f"Otsu自動閾値: {int(t)}\n赤マスクを見て微調整してください。")

    # -- ライブプレビュー -------------------------------------------
    def _on_preview_toggle(self):
        if self.v_preview.get():
            self.update_preview()
        else:
            self._clear_overlay()

    def _schedule_preview(self):
        """スライダー連続変化をデバウンスしてプレビュー更新(重い再計算を間引く)。"""
        if self.frame is None:
            return
        # 解析結果を表示中にパラメータを触ったら、ライブプレビューに戻す
        if self.result is not None:
            self.result = None
            self.show_frame()
        if not self.v_preview.get():
            return
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(110, self.update_preview)

    def _clear_overlay(self):
        if self._overlay_artist is not None:
            H, W = self.frame.shape[:2] if self.frame is not None else (1, 1)
            self._overlay_artist.set_data(np.zeros((H, W, 4), np.float32))
            self.canvas.draw_idle()

    def update_preview(self):
        """現在のスライダー設定でマスクを計算し、赤の半透明で重ねる。"""
        self._preview_job = None
        if self.frame is None or not self.v_preview.get():
            return
        if self.result is not None:
            # 解析結果表示中はプレビューを出さない(結果を上書きしない)
            return
        if self.roi is None:
            H, W = self.frame.shape[:2]
            self.roi = (0, 0, W, H)
        try:
            mask = wire_mask(self.frame, self.roi,
                             thr=int(self.v_thr.get()),
                             blob=int(self.v_blob.get()),
                             close=int(self.v_close.get()),
                             use_color=self.v_color.get())
        except Exception:
            return
        H, W = self.frame.shape[:2]
        overlay = np.zeros((H, W, 4), np.float32)
        overlay[mask > 0] = (1.0, 0.15, 0.15, 0.55)   # 赤・半透明
        if self._overlay_artist is None:
            self._overlay_artist = self.ax_img.imshow(overlay, zorder=5)
        else:
            self._overlay_artist.set_data(overlay)
        # 画素数を出して「今どれだけ拾っているか」を可視化
        n = int(np.count_nonzero(mask))
        self.ax_img.set_title(
            f"画像 + 抽出マスク(赤)  拾った画素={n}  — ドラッグでROI指定")
        self.canvas.draw_idle()

    # -- 表示 --------------------------------------------------------
    def show_frame(self):
        if self.frame is None:
            return
        self._loupe_bg = None        # 画像が変わったのでブリット背景を無効化
        self.ax_img.clear()
        self._overlay_artist = None
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self._img_artist = self.ax_img.imshow(rgb)
        self.ax_img.set_title("画像 + 抽出マスク(赤)  — ドラッグでROI指定")
        # プレビュー時はカラーバー枠を隠す(古い凡例を残さない)
        if getattr(self, "cax", None) is not None:
            self.cax.cla()
            self.cax.set_visible(False)
        self._draw_roi_rect()
        self.canvas.draw_idle()

    def analyze(self):
        if self.frame is None:
            messagebox.showinfo("情報", "先に動画を開いてください")
            return
        if self.roi is None:
            H, W = self.frame.shape[:2]
            self.roi = (0, 0, W, H)
        mask = wire_mask(self.frame, self.roi,
                         thr=int(self.v_thr.get()),
                         blob=int(self.v_blob.get()),
                         close=int(self.v_close.get()),
                         use_color=self.v_color.get())
        res = curvature_of(mask, smooth_scale=self.v_smooth.get() / 1000.0)
        if res is None:
            self.result = None
            self._diagnose_failure(mask)
            return
        self.result = res
        self._plot()
        self._report()

    def _diagnose_failure(self, mask):
        """抽出失敗の原因を推定して具体的なヒントを出す(直感的トラブルシュート)。"""
        n = int(np.count_nonzero(mask))
        if n == 0:
            msg = ("⚠ ワイヤを1画素も拾えませんでした。\n"
                   "→ ①明るさしきい値を下げる、または『ワイヤをクリックで調整』\n"
                   "→ ROIがワイヤを含んでいるか確認")
        elif n < 120:
            hint = ("→ 『青背景を色で分離』をONにする(細い/暗いワイヤに有効)\n"
                    if not self.v_color.get() else "")
            msg = ("⚠ 拾えた画素が少なすぎます(短い/細切れ)。\n"
                   f"  拾った画素={n}\n"
                   + hint +
                   "→ ②太い部分を消す を下げる(ワイヤまで消えている可能性)\n"
                   "→ ③線をつなぐ を上げて断片を橋渡し")
        else:
            msg = ("⚠ 画素はあるが中心線がつながりません。\n"
                   f"  拾った画素={n}\n"
                   "→ ③線をつなぐ を上げる\n"
                   "→ ROIを曲がった区間だけに絞ると安定します")
        self._log(msg)
        # 失敗時もマスクを見せて原因把握を助ける
        self.update_preview()

    # -- 解析範囲トリム(縦棒)ヘルパ ---------------------------------
    def _inc_range_px(self, res):
        """トリムで選ばれた弧長範囲[px]を返す。"""
        total = res["s"][-1]
        return self.s_lo_frac * total, self.s_hi_frac * total

    def _inc_mask(self, res):
        """採用区間の真偽マスク。絞りすぎ(3点未満)は全域扱いに戻す。"""
        lo, hi = self._inc_range_px(res)
        m = (res["s"] >= lo) & (res["s"] <= hi)
        if int(m.sum()) < 3:
            m = np.ones_like(res["s"], dtype=bool)
        return m

    def _on_metric_toggle(self):
        """κ ↔ R 表示切替。解析結果があれば描き直す。"""
        if self.result is not None:
            self._plot()
            self._report()

    def _kappa_signed(self, res):
        """符号付きトグルに応じた曲率配列を返す(OFFなら絶対値)。"""
        k = res["kappa"]
        return k if self.v_signed.get() else np.abs(k)

    def _metric_disp(self, res):
        """表示モード(κ/R)に応じた値配列と単位ラベルを返す。
        返り値: (値配列, 名前, 単位, is_radius)。px_per_mm があれば実寸単位。
        符号付きモードでは負値(曲がる向き)も保持する。"""
        k = self._kappa_signed(res)
        if self.v_radius.get():
            with np.errstate(divide="ignore", invalid="ignore"):
                R = 1.0 / k                       # 曲率半径 R = 1/κ [px](符号継承)
            if self.px_per_mm:
                return R / self.px_per_mm, "R", "mm", True
            return R, "R", "px", True
        if self.px_per_mm:
            return k * self.px_per_mm, "κ", "1/mm", False
        return k, "κ", "1/px", False

    @staticmethod
    def _disp_limits(vals, is_radius):
        """カラースケール/縦軸の頑健な範囲を返す(外れ値でスケールが潰れないよう)。
        両符号あれば 0 を中心とした対称範囲。R は直線部で発散するので
        絶対値のパーセンタイルで上限を抑える。"""
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return 0.0, 1.0
        pct = 95 if is_radius else 97
        hi = float(np.nanpercentile(np.abs(v), pct))
        if not np.isfinite(hi) or hi <= 0:
            hi = float(np.nanmax(np.abs(v))) or 1.0
        neg, pos = bool(np.any(v < 0)), bool(np.any(v > 0))
        if neg and pos:
            return -hi, hi          # 両符号 → 0中心の対称スケール
        if neg:
            return -hi, 0.0
        return 0.0, hi

    def _plot(self):
        res = self.result
        v_disp, name, unit, is_r = self._metric_disp(res)
        inc = self._inc_mask(res)
        vmin, vmax = self._disp_limits(v_disp[inc], is_r)

        # 画像 + カラー中心線(除外部は薄い灰色)
        self.ax_img.clear()
        self._overlay_artist = None
        self.ax_img.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        if (~inc).any():
            self.ax_img.scatter(res["x"][~inc], res["y"][~inc],
                                c="0.7", s=5, alpha=0.5)
        # 両符号(0中心)なら発散カラーマップ、片符号なら従来の連続カラーマップ
        # (R は「小さい=急な曲がり」を目立たせたいので反転)
        if vmin < 0 < vmax:
            cmap = "coolwarm"
        else:
            cmap = "jet_r" if is_r else "jet"
        sc = self.ax_img.scatter(res["x"][inc], res["y"][inc], c=v_disp[inc],
                                 cmap=cmap, s=7, vmin=vmin, vmax=vmax)
        x0, y0, x1, y1 = self.roi
        self.ax_img.set_xlim(x0 - 10, x1 + 10)
        self.ax_img.set_ylim(y1 + 10, y0 - 10)
        label = "曲率半径" if is_r else "曲率"
        self.ax_img.set_title(f"中心線 {label}カラー  (R中央値="
                              f"{self._Rmed_str()})")
        # 固定枠(cax)にカラーバーを描き直す。ax_img のサイズは変わらない
        self.cax.set_visible(True)
        self.cax.cla()
        self.cbar = self.fig.colorbar(sc, cax=self.cax,
                                      label=f"{name} [{unit}]")
        self._draw_roi_rect()

        self._draw_kappa()
        self.canvas.draw_idle()

    def _draw_kappa(self):
        """下段の分布(κ または R)を描画(トリム縦棒・除外シェード・中央値・縦軸を反映)。
        ドラッグ中はここだけ呼んで軽く追従させる。"""
        res = self.result
        if res is None:
            return
        v_disp, name, unit, is_r = self._metric_disp(res)
        s_disp = res["s"] / self.px_per_mm if self.px_per_mm else res["s"]
        su = "mm" if self.px_per_mm else "px"
        inc = self._inc_mask(res)
        vmin, vmax = self._disp_limits(v_disp[inc], is_r)
        total = s_disp[-1]
        lo_d, hi_d = self.s_lo_frac * total, self.s_hi_frac * total

        self.ax_k.clear()
        # 両符号なら 0 の基準線を引く
        if vmin < 0 < vmax:
            self.ax_k.axhline(0, color="0.4", lw=0.8)
        # 全体は薄灰、採用区間だけ濃い青で強調
        self.ax_k.plot(s_disp, v_disp, color="0.75", lw=1.0)
        self.ax_k.plot(s_disp[inc], v_disp[inc], "b-", lw=1.7)
        if self.s_lo_frac > 0:
            self.ax_k.axvspan(s_disp[0], lo_d, color="0.5", alpha=0.15)
        if self.s_hi_frac < 1:
            self.ax_k.axvspan(hi_d, s_disp[-1], color="0.5", alpha=0.15)
        med = np.nanmedian(v_disp[inc])
        self.ax_k.axhline(med, color="r", ls="--",
                          label=f"中央値={med:.4g} {unit}")
        # ドラッグ可能な緑の縦棒(左右)
        l_lo = self.ax_k.axvline(lo_d, color="#0a0", lw=2.2, alpha=0.9)
        l_hi = self.ax_k.axvline(hi_d, color="#0a0", lw=2.2, alpha=0.9)
        self._trim_lines = [l_lo, l_hi]
        self.ax_k.set_xlabel(
            f"弧長 s [{su}]  — 緑の縦棒をドラッグで範囲指定 / ダブルクリックで全域")
        self.ax_k.set_ylabel(f"{name} [{unit}]")
        # 縦軸範囲: 符号に応じて 0 中心 / 片側
        ylo = vmin * 1.3 if vmin < 0 else 0.0
        yhi = vmax * 1.3 if vmax > 0 else 0.0
        if yhi <= ylo:
            yhi = ylo + 1.0
        self.ax_k.set_ylim(ylo, yhi)
        # 横軸範囲: トリム済みなら採用区間にズーム(端の長い直線区間で
        # 肝心の曲がりが潰れるのを防ぐ)。ドラッグ中は全体を出して位置決めしやすく。
        trimmed = self.s_lo_frac > 0 or self.s_hi_frac < 1
        if trimmed and self._drag_idx is None:
            span = max(hi_d - lo_d, 1e-6)
            pad = 0.10 * span            # 縦棒を掴む余白 + 少しの文脈
            self.ax_k.set_xlim(max(s_disp[0], lo_d - pad),
                               min(s_disp[-1], hi_d + pad))
        else:
            self.ax_k.set_xlim(s_disp[0], s_disp[-1])
        self.ax_k.grid(alpha=0.3)
        self.ax_k.legend(loc="upper right")
        self.ax_k.set_title("局所曲率半径 R(s)" if is_r else "局所曲率 κ(s)")

    # -- 縦棒ドラッグ ------------------------------------------------
    def _on_trim_press(self, ev):
        if ev.inaxes != self.ax_k or self.result is None or ev.xdata is None:
            return
        if getattr(ev, "dblclick", False):     # ダブルクリックで全域に戻す
            self.s_lo_frac, self.s_hi_frac = 0.0, 1.0
            self._drag_idx = None
            self._plot(); self._report()
            return
        if not self._trim_lines:
            return
        x0, x1 = self.ax_k.get_xlim()
        tol = 0.03 * (x1 - x0)
        lo_x = self._trim_lines[0].get_xdata()[0]
        hi_x = self._trim_lines[1].get_xdata()[0]
        d_lo, d_hi = abs(ev.xdata - lo_x), abs(ev.xdata - hi_x)
        if min(d_lo, d_hi) > tol:
            return
        self._drag_idx = 0 if d_lo <= d_hi else 1

    def _on_trim_motion(self, ev):
        if self._drag_idx is None or ev.inaxes != self.ax_k or ev.xdata is None:
            return
        total = self.result["s"][-1] / (self.px_per_mm or 1.0)
        if total <= 0:
            return
        frac = float(np.clip(ev.xdata / total, 0.0, 1.0))
        if self._drag_idx == 0:
            self.s_lo_frac = max(0.0, min(frac, self.s_hi_frac - 0.02))
        else:
            self.s_hi_frac = min(1.0, max(frac, self.s_lo_frac + 0.02))
        self._draw_kappa()          # 下グラフだけ即時更新(軽量)
        self.canvas.draw_idle()

    def _on_trim_release(self, ev):
        if self._drag_idx is None:
            return
        self._drag_idx = None
        self._plot()                # 離したら上の画像も再カラー
        self._report()

    def _Rmed_str(self):
        res = self.result
        inc = self._inc_mask(res)
        Rpx = 1.0 / np.nanmedian(self._kappa_signed(res)[inc])
        if self.px_per_mm:
            return f"{Rpx/self.px_per_mm:.2f} mm"
        return f"{Rpx:.1f} px"

    def _report(self):
        res = self.result
        inc = self._inc_mask(res)
        k = self._kappa_signed(res)[inc]
        kmed = np.nanmedian(k)
        kmean = np.nanmean(k)
        Rpx = 1.0 / kmed
        s_inc = res["s"][inc]
        seg_len = float(s_inc[-1] - s_inc[0])
        trimmed = self.s_lo_frac > 0 or self.s_hi_frac < 1
        lines = [f"[フレーム {self.cur_index}]"]
        if trimmed:
            lines.append(f" 採用範囲: {self.s_lo_frac*100:.0f}–"
                         f"{self.s_hi_frac*100:.0f}% (端ノイズ除外)")
        lines.append(f" 弧長  : {seg_len:.1f} px"
                     + ("(採用区間)" if trimmed else ""))
        if self.px_per_mm:
            lines += [
                f"       = {seg_len/self.px_per_mm:.2f} mm",
                f" κ中央値: {kmed*self.px_per_mm:.5f} 1/mm",
                f" κ平均 : {kmean*self.px_per_mm:.5f} 1/mm",
                f" R中央値: {Rpx/self.px_per_mm:.2f} mm"]
        else:
            lines += [
                f" κ中央値: {kmed:.5f} 1/px",
                f" κ平均 : {kmean:.5f} 1/px",
                f" R中央値: {Rpx:.1f} px",
                " (未校正: 実寸は校正後に表示)"]
        self._log("\n".join(lines))

    # -- 出力 --------------------------------------------------------
    def _outdir(self):
        return os.path.dirname(os.path.abspath(self.video_path or __file__))

    def export_frame_csv(self):
        if self.result is None:
            messagebox.showinfo("情報", "先に『解析』してください")
            return
        res = self.result
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialdir=self._outdir(),
            initialfile=f"curvature_frame{self.cur_index}.csv")
        if not path:
            return
        ks = self._kappa_signed(res)             # 符号付き/絶対値トグルを反映
        if self.px_per_mm:
            s = res["s"] / self.px_per_mm
            k = ks * self.px_per_mm
            xmm = res["x"] / self.px_per_mm
            ymm = res["y"] / self.px_per_mm
            hdr = "s_mm,x_mm,y_mm,kappa_1permm,R_mm"
            data = np.column_stack([s, xmm, ymm, k, 1.0 / k])
        else:
            hdr = "s_px,x_px,y_px,kappa_1perpx,R_px"
            data = np.column_stack([res["s"], res["x"], res["y"],
                                    ks, 1.0 / ks])
        np.savetxt(path, data, delimiter=",", header=hdr, comments="",
                   fmt="%.6g")
        self._log(f"CSV保存: {os.path.basename(path)}")

    def save_annotated(self):
        if self.result is None:
            messagebox.showinfo("情報", "先に『解析』してください")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialdir=self._outdir(),
            initialfile=f"annotated_frame{self.cur_index}.png")
        if path:
            self.fig.savefig(path, dpi=130)
            self._log(f"画像保存: {os.path.basename(path)}")

    def batch_all(self):
        if self.cap is None:
            messagebox.showinfo(
                "情報", "全フレーム一括は動画用の機能です。\n"
                        "静止画では『解析』『このフレームCSV書出し』をお使いください。")
            return
        step = simpledialog.askinteger(
            "一括解析", "何フレームおきに解析しますか?\n(例: 5)",
            initialvalue=5, minvalue=1)
        if not step:
            return
        thr = int(self.v_thr.get()); blob = int(self.v_blob.get())
        close = int(self.v_close.get()); sm = self.v_smooth.get() / 1000.0
        use_color = self.v_color.get()
        rows = []
        idxs = range(0, self.nframes, step)
        self._log(f"一括解析中… ({len(list(idxs))}フレーム)")
        self.root.update()
        for i in range(0, self.nframes, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = self.cap.read()
            if not ok:
                continue
            m = wire_mask(fr, self.roi, thr=thr, blob=blob, close=close,
                          use_color=use_color)
            r = curvature_of(m, smooth_scale=sm)
            if r is None:
                continue
            # 各フレームにも同じトリム割合を適用(端ノイズを一括で除外)
            inc = self._inc_mask(r)
            s_inc = r["s"][inc]
            seg_len = float(s_inc[-1] - s_inc[0])
            kmed = float(np.nanmedian(self._kappa_signed(r)[inc]))
            rows.append((i, i / self.fps, seg_len, kmed, 1.0 / kmed))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_index)
        self.cap.read()
        if not rows:
            self._log("⚠ 一括解析で有効な結果がありません")
            return
        arr = np.array(rows)
        # 換算
        if self.px_per_mm:
            length = arr[:, 2] / self.px_per_mm
            kappa = arr[:, 3] * self.px_per_mm
            R = arr[:, 4] / self.px_per_mm
            lu, ku, ru = "mm", "1/mm", "mm"
        else:
            length, kappa, R = arr[:, 2], arr[:, 3], arr[:, 4]
            lu, ku, ru = "px", "1/px", "px"
        out = os.path.join(self._outdir(), "curvature_timeseries.csv")
        np.savetxt(out,
                   np.column_stack([arr[:, 0], arr[:, 1], length, kappa, R]),
                   delimiter=",",
                   header=f"frame,time_s,arclength_{lu},kappa_med_{ku},R_med_{ru}",
                   comments="", fmt="%.6g")
        # グラフ(上段は κ/R トグルに追従。CSVは常に両方を保存)
        is_r = self.v_radius.get()
        y_top = R if is_r else kappa
        yl_top = f"R中央値 [{ru}]" if is_r else f"κ中央値 [{ku}]"
        ti_top = ("送り(時間)に対する中央曲率半径" if is_r
                  else "送り(時間)に対する中央曲率")
        win = tk.Toplevel(self.root)
        win.title("時系列: 曲率半径と弧長" if is_r else "時系列: 曲率と弧長")
        fig = Figure(figsize=(8, 6))
        a1 = fig.add_subplot(2, 1, 1)
        a1.plot(arr[:, 1], y_top, "o-", ms=3)
        a1.set_ylabel(yl_top); a1.grid(alpha=.3)
        a1.set_title(ti_top)
        a2 = fig.add_subplot(2, 1, 2)
        a2.plot(arr[:, 1], length, "s-", ms=3, color="green")
        a2.set_xlabel("時間 [s]"); a2.set_ylabel(f"弧長 [{lu}]")
        a2.grid(alpha=.3)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack(fill="both", expand=True)
        if is_r:
            self._log(f"一括解析完了: {len(rows)}点\n保存: {os.path.basename(out)}\n"
                      f"R中央値 平均={np.nanmean(R):.4g} {ru}")
        else:
            self._log(f"一括解析完了: {len(rows)}点\n保存: {os.path.basename(out)}\n"
                      f"κ中央値 平均={np.nanmean(kappa):.4g} {ku}")

    # -- util --------------------------------------------------------
    def _log(self, msg):
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", msg)


def main():
    root = tk.Tk()
    try:
        root.state("zoomed")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
