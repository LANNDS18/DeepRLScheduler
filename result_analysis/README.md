# 📊 `result_analysis` 

Dive deep into the results and analysis components of our project in this directory.

## 📁 Contents

- **📈 `csv_logs`**: Inside this directory, you'll find CSV log files. They are systematically arranged based on the experiment order outlined in our dissertation. Whether they originate from tensorboard log downloads, priority-based method evaluations, or RL method evaluations, they're all housed here.

    To download tensorboard log into csv files, you should run:

    ```bash
    tensorboard --logdir <YOUR TENSORBOARD-LOG DIRECTORIES>
    ```

- **📚 `trace_data_visualization.ipynb`**: A Jupyter notebook dedicated to data analysis. Dive in to explore visual representations of data distribution, percentiles, and density specific to our dataset in this project.

- **📉 `training_log_visualization.ipynb`**: While TensorBoard offers a quick look at training logs, this Jupyter notebook takes it a notch higher. It gives us the flexibility to contrast, compare, and draw deeper insights from our training logs.

