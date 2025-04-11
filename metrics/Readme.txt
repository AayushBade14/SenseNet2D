ModelMetricsEvaluator – quick & easy model evaluation 📊
-------------------------------------------------------

This little class is something I put together to make life easier when checking how well a classification model is doing. Instead of writing the same few lines over and over (accuracy, confusion matrix, etc.), this handles it all in one shot.

What it does:
-------------
- Shows accuracy
- Prints out the full classification report (precision, recall, F1, etc.)
- Displays the confusion matrix in text AND as a nice little plot (if you want)
- Returns all of that as a dictionary so you can use it elsewhere if needed

Stuff you'll need:
------------------
- `scikit-learn`
- `matplotlib`

If you don’t already have those:

    pip install scikit-learn matplotlib

How to use it:
--------------
Just do something like this:

```python
evaluator = ModelMetricsEvaluator(model_name="My Cool Model")
evaluator.evaluate(y_true, y_pred, display_plot=True)
