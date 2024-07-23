import json

tracenorm_orthogonal_mtt = json.load(open("models/mr_Cuong_grid_search/multitask_LSTM/0.4-0.9-0.3-0.001/history_training.json", "r"))
tracenorm_orthogonal_mtt = tracenorm_orthogonal_mtt["train"]

orthogonal_mtt = json.load(open("models/mr_Dieu_grid_search/w_regression-0.1/w_classify-0.5/w_grad-0.5/LSTM_n_hidden1-_n_hidden2-_p_dropout-_-learning_rate_w_regression-0.1_w_classify-0.5_w_grad-0.5/history_training.json", "r"))
orthogonal_mtt = orthogonal_mtt["train"]

mtt = json.load(open("models/multitask_LSTM_base/history_training.json", 'r'))
mtt = mtt['train']

cross_entropy_losses = {
    "Multitask + Orthogonal + Tracenorm": tracenorm_orthogonal_mtt["Train Loss Cls"],
    "Multitask + Orthogonal": orthogonal_mtt["Train Loss Cls"],
    "Multitask": mtt["Train Loss Cls"]
}

import plotly.graph_objects as go

# Assuming you have a dictionary containing loss names and their values
def visualize_training_losses(loss_data):
  """
  This function creates a Plotly figure visualizing multiple training losses.

  Args:
      loss_data: A dictionary where keys are loss names (strings) and values are 
                  lists of loss values (floats).

  Returns:
      A Plotly figure object containing the visualization.
  """
  plots = []  # List to store individual loss trace objects

  for loss_name, loss_values in loss_data.items():
    epochs = list(range(len(loss_values)))  # Assuming loss has same length as epochs
    plot = go.Scatter(
        x=epochs,
        y=loss_values,
        name=loss_name
    )
    plots.append(plot)

  layout = go.Layout(
      xaxis_title='Epochs',
      yaxis_title='Training Loss',
      title='Training Loss Comparison'
  )

  fig = go.Figure(data=plots, layout=layout)
  return fig

visualize_training_losses(
  loss_data=cross_entropy_losses
)