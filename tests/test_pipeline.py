import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import pipeline
from pipeline import baseline_predict, clean, neighbour_price


def raw(**over):
    row = {"price": 200_000, "date": "2024-03-15 00:00", "postcode": "B15 2TT", "type": "T", "new_build": "N",
           "tenure": "F", "district": "BIRMINGHAM", "county": "WEST MIDLANDS", "category": "A"}
    return {**row, **over}


def test_clean_drops_non_standard_sales_and_derives_fields():
    df = clean(pd.DataFrame([raw(), raw(category="B"), raw(type="O"), raw(price=5_000_000), raw(postcode=None)]))
    assert len(df) == 1
    assert df.outcode.iloc[0] == "B15" and df.month_idx.iloc[0] == 2


def test_baseline_falls_back_from_outcode_to_type_to_overall():
    train = pd.DataFrame({"outcode": ["B1", "B1", "B2"], "type": ["T", "T", "F"], "price": [100, 300, 500]})
    test = pd.DataFrame({"outcode": ["B1", "B9", "B9"], "type": ["T", "F", "D"]})
    assert list(baseline_predict(train, test)) == [200, 500, 300]


def test_neighbour_price_leaves_out_the_sale_itself(monkeypatch):
    monkeypatch.setattr(pipeline, "K", 2)
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 5.0]])  # first two share a postcode
    logp = np.array([10.0, 20.0, 30.0, 40.0])
    nn = NearestNeighbors().fit(pts)
    assert list(neighbour_price(nn, logp, pts, exclude_self=True)) == [25.0, 20.0, 15.0, 25.0]
    assert neighbour_price(nn, logp, np.array([[0.0, 4.9]]), exclude_self=False)[0] == 35.0
