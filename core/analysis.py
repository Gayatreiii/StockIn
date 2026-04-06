import numpy as np
import pandas as pd

# ── Try importing PyTorch (required for deep learning models) ─────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

# ── 1. Attention Mechanism ────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    """
    Bahdanau-style attention. Learns which time steps matter most
    for prediction. Used inside LSTM and Hybrid models.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out).squeeze(-1)           # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)          # (batch, hidden)
        return context


# ── 2. Bidirectional LSTM + Attention ────────────────────────────────────────
class LSTMAttentionModel(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism.
    Processes stock price sequences in both directions,
    then focuses on the most informative time steps.
    """
    def __init__(self, input_size: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.attention = AttentionLayer(hidden * 2)  # *2 for bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = self.attention(out)
        return self.fc(ctx)


# ── 3. CNN for Temporal Pattern Detection ────────────────────────────────────
class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network.
    Treats the price sequence like an image — detects local patterns
    (e.g., head & shoulders, double bottom) using convolutional filters.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x: (batch, seq, features) → (batch, features, seq) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        return self.fc(x)


# ── 4. Positional Encoding (for Transformer) ─────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── 5. Transformer with Multi-Head Self-Attention ────────────────────────────
class TransformerModel(nn.Module):
    """
    Transformer encoder with positional encoding and multi-head attention.
    Self-attention lets each day attend to all other days simultaneously —
    captures long-range dependencies in price history.
    """
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)   # global average over sequence
        return self.fc(x)


# ── 6. Hybrid CNN-LSTM with Attention ────────────────────────────────────────
class HybridCNNLSTM(nn.Module):
    """
    Hybrid Deep Learning: CNN extracts local patterns → LSTM learns
    temporal dependencies → Attention focuses on important steps.
    This is the main prediction model.
    """
    def __init__(self, input_size: int, hidden: int = 64):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # LSTM sequence model
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True, dropout=0.3)
        # Attention
        self.attention = AttentionLayer(hidden)
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # CNN: (batch, seq, feat) → (batch, feat, seq) → CNN → (batch, 64, seq)
        cnn_out = self.cnn(x.permute(0, 2, 1))
        # Back to (batch, seq, 64) for LSTM
        lstm_out, _ = self.lstm(cnn_out.permute(0, 2, 1))
        ctx = self.attention(lstm_out)
        return self.fc(ctx)


# ── 7. Graph Neural Network (GNN) for indicator relationships ─────────────────
class GNNLayer(nn.Module):
    """
    Graph Attention Network layer.
    Each technical indicator is a node. The GNN learns which
    indicators are most correlated and weights them accordingly.
    """
    def __init__(self, in_feat: int, out_feat: int):
        super().__init__()
        self.W = nn.Linear(in_feat, out_feat, bias=False)
        self.a = nn.Linear(2 * out_feat, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        h = self.W(x)           # (N, out_feat)
        N = h.size(0)
        hi = h.unsqueeze(1).expand(-1, N, -1)
        hj = h.unsqueeze(0).expand(N, -1, -1)
        e = self.leaky(self.a(torch.cat([hi, hj], dim=-1)).squeeze(-1))
        mask = -9e15 * torch.ones_like(e)
        attn = torch.where(adj > 0, e, mask)
        attn = torch.softmax(attn, dim=1)
        return torch.relu(torch.matmul(attn, h))


class GNNSignal(nn.Module):
    """Two-layer GNN for technical indicator graph."""
    def __init__(self, n_nodes: int):
        super().__init__()
        self.gnn1 = GNNLayer(1, 16)
        self.gnn2 = GNNLayer(16, 8)
        self.out = nn.Linear(n_nodes * 8, 3)   # bullish / neutral / bearish

    def forward(self, x, adj):
        x = self.gnn1(x, adj)
        x = self.gnn2(x, adj)
        return self.out(x.view(1, -1))


# ════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

FEATURES = ["Price_Change", "Volume_Change", "RSI", "MA10", "MA20", "MA50",
            "Volatility", "BB_Upper_Pct", "BB_Lower_Pct", "Momentum5"]
SEQ_LEN = 40


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features needed by deep learning models."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    out["Price_Change"] = close.pct_change()
    out["Volume_Change"] = df["Volume"].pct_change()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # Moving averages (normalised by close)
    out["MA10"] = close.rolling(10).mean() / (close + 1e-8)
    out["MA20"] = close.rolling(20).mean() / (close + 1e-8)
    out["MA50"] = close.rolling(min(50, len(close))).mean() / (close + 1e-8)

    # Volatility (rolling std of returns)
    out["Volatility"] = out["Price_Change"].rolling(20).std()

    # Bollinger band position
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    out["BB_Upper_Pct"] = (bb_upper - close) / (close + 1e-8)
    out["BB_Lower_Pct"] = (close - bb_lower) / (close + 1e-8)

    # 5-day momentum
    out["Momentum5"] = close.pct_change(5)

    # Target: 1 if next day is up
    out["Target"] = (close.shift(-1) > close).astype(int)

    return out.dropna()


def _make_sequences(feat_df: pd.DataFrame):
    """Convert feature dataframe to (X, y) sequences for deep learning."""
    feat_cols = [c for c in FEATURES if c in feat_df.columns]
    data = feat_df[feat_cols].values.astype(np.float32)
    targets = feat_df["Target"].values.astype(np.int64)

    # Normalise
    mu = data.mean(axis=0)
    sigma = data.std(axis=0) + 1e-8
    data = (data - mu) / sigma

    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i: i + SEQ_LEN])
        y.append(targets[i + SEQ_LEN])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), len(feat_cols)


def _train(model, X_tr, y_tr, epochs: int = 20, lr: float = 1e-3):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def _predict_one(model, X_last):
    """Get prediction and confidence for the last sequence."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_last).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return ("UP" if probs[1] > 0.5 else "DOWN"), round(float(max(probs)) * 100, 1)


def _accuracy(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te)).argmax(dim=1).numpy()
    return round(float((preds == y_te).mean() * 100), 1)


# ════════════════════════════════════════════════════════════════════════════
# GNN SIGNAL
# ════════════════════════════════════════════════════════════════════════════

def _gnn_signal(df: pd.DataFrame) -> tuple[float, str]:
    """
    Run GNN over technical indicator nodes to get a graph-level signal.
    Returns (score -1..1, direction).
    """
    if not TORCH_AVAILABLE or df is None or len(df) < 20:
        return 0.0, "NEUTRAL"

    close = df["Close"]
    indicators = {}

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
    indicators["rsi_norm"] = float((rsi.iloc[-1] - 50) / 50)

    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    indicators["ma_cross"] = float((ma10 - ma20) / (close.iloc[-1] + 1e-8))

    if len(close) >= 50:
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(min(200, len(close))).mean().iloc[-1]
        indicators["golden_cross"] = float((ma50 - ma200) / (close.iloc[-1] + 1e-8))

    if len(close) >= 5:
        indicators["momentum5"] = float(
            (close.iloc[-1] - close.iloc[-5]) / (close.iloc[-5] + 1e-8)
        )

    vol = df["Volume"].pct_change().fillna(0)
    indicators["vol_change"] = float(np.clip(vol.iloc[-1], -2, 2))

    returns = close.pct_change().dropna()
    rv = returns.tail(5).std()
    nv = returns.tail(20).std()
    indicators["vol_regime"] = float(-(rv / (nv + 1e-8) - 1))

    vals = list(indicators.values())
    n = len(vals)
    if n < 3:
        return 0.0, "NEUTRAL"

    try:
        x = torch.tensor([[v] for v in vals], dtype=torch.float32)
        adj = torch.ones(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = 0.5 + 0.5 * float(np.sign(vals[i] * vals[j]))

        model = GNNSignal(n_nodes=n)
        model.eval()
        with torch.no_grad():
            out = model(x, adj)
            probs = torch.softmax(out, dim=1)[0].numpy()
        score = float(probs[0] - probs[2])    # bullish - bearish
        direction = "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL"
        return score, direction
    except Exception:
        score = sum(1 if v > 0 else -1 for v in vals) / n
        direction = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
        return score, direction


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API  — same interface as before so app.py needs no changes
# ════════════════════════════════════════════════════════════════════════════

def get_technical_indicators(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 20:
        return {}
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(min(50, len(close))).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    return {
        "RSI":      round(float(rsi.iloc[-1]), 2),
        "MA10":     round(float(ma10.iloc[-1]), 2),
        "MA20":     round(float(ma20.iloc[-1]), 2),
        "MA50":     round(float(ma50.iloc[-1]), 2),
        "BB_Upper": round(float(bb_upper.iloc[-1]), 2),
        "BB_Lower": round(float(bb_lower.iloc[-1]), 2),
    }


def get_support_resistance(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 20:
        return {}
    close = df["Close"]
    current = float(close.iloc[-1])
    resistance = float(df["High"].tail(20).max())
    support = float(df["Low"].tail(20).min())
    return {
        "current":    round(current, 2),
        "resistance": round(resistance, 2),
        "support":    round(support, 2),
    }


def simple_prediction(df: pd.DataFrame) -> dict:
    """
    Deep learning ensemble prediction.
    Uses: Hybrid CNN-LSTM + Attention, LSTM + Attention,
          CNN, Transformer, GNN.
    Falls back to rule-based if torch not available or data too short.
    """
    # ── Fallback: rule-based if torch not installed or data too short ─────────
    if not TORCH_AVAILABLE or df is None or len(df) < SEQ_LEN + 10:
        return _rule_based_fallback(df)

    # ── Build features & sequences ────────────────────────────────────────────
    try:
        feat_df = _build_features(df)
    except Exception:
        return _rule_based_fallback(df)

    if len(feat_df) < SEQ_LEN + 10:
        return _rule_based_fallback(df)

    X, y, n_feat = _make_sequences(feat_df)
    split = max(int(len(X) * 0.8), 1)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # ── Define all deep learning models ───────────────────────────────────────
    model_defs = {
        "Hybrid CNN-LSTM": HybridCNNLSTM(input_size=n_feat),
        "LSTM+Attention":  LSTMAttentionModel(input_size=n_feat),
        "CNN":             CNNModel(input_size=n_feat),
        "Transformer":     TransformerModel(input_size=n_feat),
    }

    all_results = {}
    votes_up = 0
    total_conf = 0.0

    for name, model in model_defs.items():
        try:
            trained = _train(model, X_tr, y_tr, epochs=20)
            pred_label, conf = _predict_one(trained, X[-1])
            acc = _accuracy(trained, X_te, y_te) if len(X_te) > 0 else 50.0
            if pred_label == "UP":
                votes_up += 1
            total_conf += conf
            all_results[name] = {
                "prediction": pred_label,
                "confidence": conf,
                "accuracy": acc,
            }
        except Exception:
            all_results[name] = {"prediction": "HOLD", "confidence": 50.0, "accuracy": 50.0}

    # ── GNN signal ────────────────────────────────────────────────────────────
    gnn_score, gnn_dir = _gnn_signal(df)
    if gnn_dir == "BULLISH":
        votes_up += 1
    all_results["GNN"] = {
        "prediction": "UP" if gnn_dir == "BULLISH" else "DOWN" if gnn_dir == "BEARISH" else "HOLD",
        "confidence": round(abs(gnn_score) * 100, 1),
        "accuracy": 0.0,
    }

    total_models = len(model_defs) + 1  # +1 for GNN
    final = "BUY" if votes_up > total_models / 2 else "SELL" if votes_up < total_models / 2 else "HOLD"
    avg_conf = round(total_conf / len(model_defs), 1)

    # Build reason string
    up_count = votes_up
    dn_count = total_models - votes_up
    reason = (
        f"Hybrid CNN-LSTM · LSTM+Attn · CNN · Transformer · GNN — "
        f"{up_count}/{total_models} models bullish"
    )

    return {
        "direction":   final,
        "confidence":  avg_conf,
        "reason":      reason,
        "score":       votes_up - dn_count,
        "all_models":  all_results,
        "gnn_score":   round(gnn_score, 3),
    }


# ── Rule-based fallback (original logic, unchanged) ───────────────────────────
def _rule_based_fallback(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 20:
        return {"direction": "HOLD", "confidence": 50, "reason": "Insufficient data", "score": 0}

    indicators = get_technical_indicators(df)
    close = df["Close"]
    rsi  = indicators.get("RSI", 50)
    ma10 = indicators.get("MA10", close.iloc[-1])
    ma20 = indicators.get("MA20", close.iloc[-1])
    current = float(close.iloc[-1])
    score, reasons = 0, []

    if rsi < 35:
        score += 2; reasons.append(f"RSI {rsi:.0f} oversold")
    elif rsi > 65:
        score -= 2; reasons.append(f"RSI {rsi:.0f} overbought")
    elif rsi > 50:
        score += 1; reasons.append(f"RSI {rsi:.0f} bullish zone")
    else:
        score -= 1; reasons.append(f"RSI {rsi:.0f} bearish zone")

    if ma10 > ma20:
        score += 2; reasons.append("MA10 > MA20 uptrend")
    else:
        score -= 2; reasons.append("MA10 < MA20 downtrend")

    if current > ma20:
        score += 1; reasons.append("Price above MA20")
    else:
        score -= 1; reasons.append("Price below MA20")

    if len(close) >= 5:
        mom = (current - float(close.iloc[-5])) / float(close.iloc[-5]) * 100
        if mom > 1.5:
            score += 1; reasons.append(f"+{mom:.1f}% momentum")
        elif mom < -1.5:
            score -= 1; reasons.append(f"{mom:.1f}% momentum")

    if score >= 3:
        direction, confidence = "BUY",  min(55 + score * 5, 82)
    elif score <= -3:
        direction, confidence = "SELL", min(55 + abs(score) * 5, 82)
    else:
        direction, confidence = "HOLD", 50 + abs(score) * 3

    return {
        "direction":  direction,
        "confidence": confidence,
        "reason":     " · ".join(reasons[:3]),
        "score":      score,
    }
