# trading_bot/tests/test_buffers.py

from trading_bot.state.book_features import BookFeatureBuffer

def test_book_feature_buffer():
    buffer = BookFeatureBuffer()

    # Test regular features
    buffer.update({'BID_DENSITY': 10, 'ASK_DENSITY': 20})
    smoothed = buffer.get_smoothed()
    assert smoothed['BID_DENSITY'] == 10

    # Test delta flow
    buffer.update_delta(buy=5, sell=2)
    buffer.update_delta(buy=3, sell=1)
    delta_flow = buffer.get_delta_flow()
    assert delta_flow != 0

    print("✅ BookFeatureBuffer basic tests passed")

if __name__ == "__main__":
    test_book_feature_buffer()