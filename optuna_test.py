from clearml.automation.optuna import OptimizerOptuna
from clearml.automation import DiscreteParameterRange, UniformParameterRange, HyperParameterOptimizer
from clearml import Task

#callback при завершении работы оптимизатора
def job_complete_callback(
    job_id,
    objective_value,
    objective_iteration,
    job_parameters,
    top_performance_job_id
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('We broke the record! Objective reached {}'.format(objective_value))

#создаем отдельную задачу для оптимизатора
Task.init(project_name='OptunaOptimization',
          task_name='optimizer',
          task_type=Task.TaskTypes.optimizer)
#оптимизатор (https://clear.ml/docs/latest/docs/references/sdk/hpo_optimization_hyperparameteroptimizer)
optimizer = HyperParameterOptimizer(
    base_task_id='061e320820a74f4c81d872943dbb84e1', #id базовой задачи, которую нужно оптимизировать
    hyper_parameters=[
                    UniformParameterRange(name='Args/tp', min_value=0.5, max_value=2, step_size=0.5),
                    UniformParameterRange(name='Args/sl', min_value=0.1, max_value=0.5, step_size=0.2),
                    DiscreteParameterRange(name='Args/n_epochs', values=[30, 50, 70]),
                ], #сетка гиперпараметров
    optimizer_class=OptimizerOptuna, #класс оптимизатора
    execution_queue='task_queue', #очередь, в которой будет выполняться задача оптимизации
    objective_metric_sign='max', #цель оптимизации (максимизация или минимизация метрики)
    objective_metric_title='Income', #название метрики
    objective_metric_series='Test income', #метка данных метрики
    max_iteration_per_job=10,
    total_max_jobs=5,
)

optimizer.start_locally(job_complete_callback=job_complete_callback)
optimizer.wait() #ждем завершения задачи оптимизации
optimizer.stop() #останавливаем задачу оптимизации
