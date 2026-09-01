# Single-egg, three-class object detection app

Place the trained model at `objdet_basic/sth_as-hw-tt.keras`, then install the
dependencies and run the Streamlit app:

```powershell
pip install -r objdet_basic/requirements.txt
streamlit run objdet_basic/app.py
```

To keep the model elsewhere, set `OBJDET_MODEL_PATH` to its full path before
starting Streamlit. The output mapping follows the original model: output 0 is
background (`sth`), output 1 is `as`, output 2 is `hw`, and output 3 is `tt`.
The app assumes that each uploaded image contains at most one egg. If several
class scores exceed the threshold, it returns only the highest-confidence class
and draws one bounding box.
