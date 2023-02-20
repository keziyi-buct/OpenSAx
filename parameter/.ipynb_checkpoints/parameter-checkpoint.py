"""
    这段代码集成了网格调参、随机调参、贝叶斯调参等调参函数
"""

from sklearn.model_selection import GridSearchCV
import hpelm
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate
import numpy as np
model_dict = {
    'pls': PLSRegression,
    'svr': SVR,
    'ann': MLPRegressor,
}



def Grid(X_train, X_test, y_train, y_test,name,param_grid):
    if name=='ann':
        MAX_ITER = 600
        model=model_dict[name](hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=MAX_ITER, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    else :
        model=model_dict[name]()
    grid = GridSearchCV(model,param_grid,cv=5)
    grid.fit(X_train, y_train)
    print("最优参数：\n", grid.best_params_, "\n")
    y_pred = grid.predict(X_test)
    y_pred=np.array(y_pred)
    np.savetxt("y_pred.txt", y_pred, fmt="%f")
    y_test=np.array(y_test)
    np.savetxt("y_test.txt", y_test, fmt="%f")
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)
    return Rmse, R2, Mae


