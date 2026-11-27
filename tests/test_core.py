from src.data_mcocr import parse_row
from src.normalize import norm_total, norm_timestamp, norm_text
from eval.metrics import evaluate, score_field


def test_parse_groups_and_fixes_labels():
    r = parse_row("A|||Tổng tiền:|||74,000|||x|||y",
                  "SELLER|||TOTAL_COST|||TOTAL_TOTAL_COST|||nan|||ADDRESS")
    assert r == {"seller": "A", "address": "y", "timestamp": None,
                 "total_cost": "Tổng tiền: 74,000"}


def test_norm_total():
    assert norm_total("54,000") == 54000
    assert norm_total("TỔNG TIỀN PHẢI T.TOÁN 1,475,000") == 1475000
    assert norm_total("HỒNG TIẾN PHẢI T.TOÁN 45.400") == 45400
    assert norm_total("191.084") == 191084
    assert norm_total("54,000.00") == 54000
    assert norm_total("Tổng:") is None
    assert norm_total(None) is None


def test_norm_timestamp():
    assert norm_timestamp("Ngày: 11/08/2020 - 12:55") == ("2020-08-11", "12:55")
    assert norm_timestamp("Ngày bán: 13/08/2020") == ("2020-08-13", None)
    assert norm_timestamp("11-08-20 8h05") == ("2020-08-11", "08:05")
    assert norm_timestamp("abc") is None


def test_norm_text_nfc_and_punct():
    decomposed = "Cà phê"
    assert norm_text(decomposed) == norm_text("Cà phê")
    assert norm_text("P. Cẩm Thủy - Quảng Ninh") == "p cẩm thủy quảng ninh"


def test_score_field_nulls():
    assert score_field("seller", None, None) == (1.0, 1.0)
    assert score_field("seller", "X", None) == (0.0, 0.0)
    assert score_field("timestamp", "11/08/2020 09:00", "Ngày: 11/08/2020 - 12:55") == (0.0, 0.5)


def test_evaluate_perfect_and_empty():
    g = {"a": {"seller": "MINIMART ANAN", "address": None,
               "timestamp": "11/08/2020 08:06", "total_cost": "74,000"}}
    perfect = {"a": {"seller": "Minimart Anan", "address": None,
                     "timestamp": "2020 khong co", "total_cost": "74.000"}}
    res = evaluate(perfect, g)
    assert res["seller"]["em"] == 1.0 and res["total_cost"]["em"] == 1.0
    assert res["timestamp"]["em"] == 0.0
    assert evaluate({}, g)["mean_em"] == 0.25  # only address (None==None) right
