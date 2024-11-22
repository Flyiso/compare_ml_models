import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class CreateModel:
    """class for creating ml/dl model from csv

    class is responsible for calling PrepareData
    to check and prepare the csv file
    and then call methods for model creation
    to create model if data is valid.
    """
    def __init__(self):
        self.dataobj = PrepareData()
        self.df = self.dataobj.verify_csv()
        self.dataobj.confirm_no_na()
        self.dataobj.choose_target()
        self.X, self.y = self.dataobj.split_dependent_independent()
        self.model_cat = self.choose_model()
        X_dummies = self.dataobj.get_x_dummies()
        y_dummies = self.dataobj.get_y_dummies(self.model_cat)
        self.dataobj.split_train_test(X_dummies, y_dummies, 0.3, 101)
        if all([self.model_cat == 'reg', self.dataobj.print_why_not_ready()]):
            self.model_type = Regressors(self.dataobj)
        elif all([self.model_cat == 'cat',
                  self.dataobj.print_why_not_ready()]):
            self.model_type = Classifiers(self.dataobj)
        else:
            self.model_type = False
            print('Try again.\n\n')
            return
        if not isinstance(self.model_type, (bool, str)):
            self.model_type.score_models()
            final_model = self.model_type.display_model_evaluation()
            self.save_model(final_model)

    def choose_model(self):
        """Method to choose if regressor or classifier"""
        print('Choose if you want a regressor or classifier')
        d_t = self.y.dtype
        print(f'Data in target column is \
        {"numerical" if d_t != "object" else "non-numerical"}')
        print(f'Data in target column has \
        {str(self.y.nunique())} unique values')
        print(f'Data in target column has \
        {str((self.y.nunique()/len(self.y))*100)} percent unique values')
        model_cat = input('\nEnter\n1 for Regressor\n2 For classifier\n')
        if model_cat == '1':
            self.model_cat = 'reg'
        elif model_cat == '2':
            self.model_cat = 'cat'
        else:
            print('Choice not valid')
            self.model_cat = False
        return self.model_cat

    def save_model(self, model):
        """save the chosen model"""
        from joblib import dump
        model_name = input('enter name to save model as:\n')
        if isinstance(model, tf.keras.models.Sequential):
            model_save = CreateANN.save_model(model_name)
            return model_save
        else:
            dump(model, f'{model_name}.joblib')
            print(f'model {model_name}.joblib is saved')
            model_save = f'{model_name}.joblib'
            return model_save


class PrepareData:
    """class responsible for preparing data
    before passing it to ml and dl classes
    """
    def __init__(self) -> None:
        """initialize"""
        self._filename = input('enter the name of csv file:\n')

    def verify_csv(self):
        """verify csv file can be found and used"""
        try:
            self.df = pd.read_csv(self._filename)
            return self.df
        except FileExistsError as error:
            print(f'{error}, cant find file at {self._filename}')
            self.df = False
            return self.df

    def confirm_no_na(self) -> bool:
        """verify no missing values"""
        df = self.df
        num_na = df.isna().sum().values.sum()
        if num_na == 0:
            self._num_na = True
            return self._num_na
        self._num_na = False
        return self._num_na

    def choose_target(self) -> str | bool:
        """verify valid target is chosen
        return target name if it is
        return False else
        """
        df = self.df
        cols = list(enumerate(df.columns))
        print('\nchoose target:')
        for col in cols:
            print(col[0], col[1])
        target = input('Enter name or Number\n')
        for pair in cols:
            if str(pair[0]) == target:
                chosen = pair[1]
                self.target = chosen
                return self.target
            elif pair[1] == target:
                chosen = pair[1]
                self.target = chosen
                return self.target
        print(f'Chosen target {target} is not a valid value')
        self.target = False
        return self.target

    def split_dependent_independent(self):
        """splits self._target from rest of df"""
        if self.target is False:
            self.X, self.y = False, False
            return self.X, self.y
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]
        return self.X, self.y

    def split_train_test(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame,
                         test_size: float, random_state: str):
        """splits to train and test if X and y are not false

        input:
        X: dataframe of df without target column
        y: series of target column
        test_size: percent size of test compared to train. float
        random_state: random state. int.

        output:
        False*4 if any input values not valid,
        X_train, X_test, y_train, y_test, splitted by test_size and
        random_state if input ok"""
        self.test_size = test_size
        self.random_state = random_state
        if all([isinstance(self.test_size, float), 0 < self.test_size < 1,
               isinstance(self.random_state, int),
               isinstance(X, pd.DataFrame),
               any([isinstance(y, pd.Series), isinstance(y, pd.DataFrame)])]):
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y,
                                 test_size=self.test_size,
                                 random_state=self.random_state)
            return self.X_train, self.X_test, self.y_train, self.y_test
        self.X_train, self.X_test = False, False
        self.y_train, self.y_test = False, False
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_y_dummies(self, mod_type):
        """get y back with dummies if needed"""
        self.y_dummies = self.y
        if mod_type is False:
            self.y_dummies = 'model selection not valid'
            return self.y_dummies
        var_limit = len(self.df)/4
        if self.y is False:
            self.y_dummies = False
        elif (str(self.y.dtype) in ['float64', 'int64']) and mod_type == 'cat':
            if self.y.nunique() > var_limit:
                print('Too many different values to convert-\
                      try regressor?')
                self.y_dummies = False
                return self.y_dummies
            self.y_dummies = pd.get_dummies(self.y)
        elif str(self.y.dtype) == 'object' and mod_type == 'cat':
            self.y_dummies = pd.get_dummies(self.y)
        elif str(self.y.dtype) == 'object' and mod_type == 'reg':
            self.y_dummies = False
        return self.y_dummies

    def get_x_dummies(self):
        """get x back with dummies if needed"""
        x_cols = list(self.X.columns)
        if 'object' not in [str(self.X[col].dtype) for col in x_cols]:
            self.X_dummies = self.X
            return self.X_dummies
        self.X_dummies = pd.get_dummies(self.X)
        return self.X_dummies

    def print_why_not_ready(self):
        """print to user why data not ready for ml"""
        self.data_ok = True
        print('\n\nProblems with data:')
        if self._num_na is False:
            print('*Dataframe has missing values')
            self.data_ok = False
        if self.target is False:
            print('*Target not valid')
            self.data_ok = False
        if not isinstance(self.random_state, int):
            print('*Random state input not valid- should be int')
            self.data_ok = False
        if not isinstance(self.test_size, float) and 0 < self._test_size < 1:
            print('*Test size was not a valid value\
                  should be integer less than 1, more than 0')
            self.data_ok = False
        if not any([isinstance(self.y_dummies, pd.Series),
                    isinstance(self.y_dummies, pd.DataFrame)]):
            if self.y_dummies == 'model selection not valid':
                print('*choice of model type was not valid')
                self.data_ok = False
            if self.y_dummies is False:
                print('*Desired model type does not match target data')
                self.data_ok = False
        if self.data_ok is True:
            print('No problems\n\ncreate model...\n')
            return True
        else:
            print('\nFix these problems and try again')
            return False


class CreateANN:
    from sklearn.preprocessing import MinMaxScaler
    """class to get attributes for ANN models"""
    def __init__(self, dataobject: PrepareData, loss: str):
        """create model
        loss options. 'mse','binary_crossentropy','categorical_crossentropy'
        """
        self.dataobject = dataobject
        from tensorflow.keras.callbacks import EarlyStopping
        self.scaler = self.MinMaxScaler()
        self.loss = loss
        self.X, self.y = self.get_data(dataobject)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.split_data(self.X, self.y, dataobject)
        self.activation = self.choose_activation()
        self.optimizer = self.choose_optimizer()
        self.batch_size = self.choose_batch_size()
        self.epochs = self.choose_epochs()
        self.multiprocess = self.choose_multiprocess()
        self.verbose = self.choose_verbose()
        self.monitor = self.choose_monitor()
        self.mode = self.choose_mode()
        self.patience = self.choose_patience()
        self.early_stop = EarlyStopping(monitor=self.monitor,
                                        mode=self.mode,
                                        patience=self.patience,
                                        verbose=self.verbose)
        if any(getattr(self, var) is False
               for var in vars(self) if var != 'multiprocess'):
            print('some input values not valid- exit model create.')
            self.model = False
            return
        else:
            self.layer_tuple = self.define_hidden_layer_tuple()
            self.model = self.build_model()

    def get_data(self, dataobject: PrepareData):
        """Method to split data and return it suitable for ann model
        """
        df_1 = dataobject.df
        self.target = dataobject.target
        self.df = pd.get_dummies(data=df_1, dtype=float)
        self.y = self.df[[col for col in self.df.columns if
                          col.startswith(self.target)]].values
        self.X = self.df[[col for col in self.df.columns if not
                          col.startswith(self.target)]].values
        if self.loss in ['binary_crossentropy', 'mse']:
            self.y = self.y[:, 0]
        return self.X, self.y

    def split_data(self, X, y, dataobject: PrepareData):
        """Split data according to loss
        send X_train and X_test to self.scale_data-
        return y_train, y_test, scaled X_train, scaled_X_test
        """
        X_train, X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=dataobject.test_size,
                             random_state=dataobject.random_state)
        self.X_train, self.X_test = self.scale_data(X_train, X_test)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self, X_train, X_test):
        """scale X data with minmax scaler
        fit transform at train data, transform test data
        return transformed train and test
        """
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def define_hidden_layer_tuple(self) -> tuple:
        """default to (100, ) if no or not valid input"""
        print('\nEnter any int to add a layer')
        print('enter negative value 0 - negative 1 for dropout layer(decimal)')
        print('enter anything else to stop adding layers')
        keep_input = True
        current_layers = []
        while keep_input:
            new_layer = input('\n')
            try:
                num = float(new_layer)
                if num <= 0 and num >= -1:
                    current_layers.append(num)
                elif num >= 1:
                    try:
                        num = int(new_layer)
                        current_layers.append(num)
                    except ValueError:
                        'No positive Floats.-'
            except ValueError:
                'exit add layers'
                keep_input = False
                print(f'current layers: {current_layers}')
        if len(current_layers) == 0:
            self.layer_tuple = (100,)
            return self.layer_tuple
        self.layer_tuple = tuple(current_layers)
        return self.layer_tuple

    def choose_activation(self) -> str:
        """default relu, if no or not valid input"""
        activation_input = self.choose_value('activation function',
                                             ['relu', 'sigmoid',
                                              'softmax', 'tanh'],
                                             'relu')
        self.activation = activation_input
        return self.activation

    def choose_optimizer(self) -> str:
        "default adam if no or not valid input"
        optimizer_input = self.choose_value('Optimizer',
                                            ['adam', 'rmsprop', 'sgd'],
                                            'adam')
        self.optimizer = optimizer_input
        return self.optimizer

    def choose_batch_size(self) -> int:
        """default 32 if no or not valid input"""
        batch_size_input = self.choose_value('batch size', default=32)
        try:
            batch_size = int(batch_size_input)
        except ValueError:
            print('input not valid- defaults to 32')
            batch_size = 32
        self.batch_size = batch_size
        return self.batch_size

    def choose_epochs(self) -> int:
        """default to 1 if no or not valid input"""
        epochs_input = self.choose_value('Epochs', default=1)
        try:
            epochs = int(epochs_input)
        except ValueError:
            print('input not valid- defaults to 1')
            epochs = 1
        self.epochs = epochs
        return self.epochs

    def choose_monitor(self) -> str:
        """default to val_loss if no or not valid input"""
        monitor_input = self.choose_value('Monitor',
                                          ['val_loss', 'accuracy'],
                                          'val_loss')
        self.monitor = monitor_input
        return self.monitor

    def choose_patience(self) -> int:
        """defaults to self.epochs if no or not valid input"""
        default_patience = self.epochs
        patience_input = self.choose_value('Patience',
                                           default=default_patience)
        try:
            patience = int(patience_input)
        except ValueError:
            patience = default_patience
        return patience

    def choose_mode(self) -> str | bool:
        """mode"""
        mode_input = self.choose_value('Mode',
                                       choices=['min', 'max', 'auto'])
        self.mode = mode_input
        return self.mode

    def choose_verbose(self) -> int | bool:
        """0-2. 0:nothing, 1:progress bar, 2:1lineperepoch"""
        verbose_input = self.choose_value('Verbose',
                                          choices=['0', '1', '2'])
        try:
            verbose = int(verbose_input)
        except ValueError:
            verbose = False
        self.verbose = verbose
        return self.verbose

    def choose_multiprocess(self) -> bool:
        """default false- use multiprocess"""
        multiprosses_input = self.choose_value(value_name='Multiprocess',
                                               choices=['True', 'False'],
                                               default='False')
        if multiprosses_input == 'True':
            self.multiprocess = True
        else:
            self.multiprocess = False
        return self.multiprocess

    def get_output_activation(self):
        """get the output activation function"""
        if self.loss == 'categorical_crossentropy':
            return 'softmax'
        elif self.loss == 'binary_crossentropy':
            return 'sigmoid'
        elif self.loss == 'mse':
            return 'linear'
        else:
            return False

    def build_model(self):
        """build Ann model"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        y_n = int(1 if len(self.y_train.shape) < 2 else self.y_train.shape[-1])
        X_n = int(1 if len(self.X_train.shape) < 2 else self.X_train.shape[-1])
        model = Sequential()
        model.add(Dense(units=X_n,
                        activation=self.activation))
        model = self.add_hidden_layers(model)
        output_funk = self.get_output_activation()
        if output_funk is False:
            print('could not match loss to output funk- stop ann creation--')
            self.model = False
            return self.model
        model.add(Dense(units=y_n,
                        activation=output_funk))
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, metrics=[self.monitor])
        self.model = model
        return self.model

    def add_hidden_layers(self, model):
        """add hidden layers to ANN model"""
        from tensorflow.keras.layers import Dense, Dropout
        model_out = model
        for value in self.layer_tuple:
            if value < 0:
                model_out.add(Dropout(abs(value), seed=101))
            else:
                model_out.add(Dense(units=value,
                                    activation=self.activation))
        return model_out

    def train_model(self):
        """train the model"""
        self.model.fit(x=self.X_train,
                       y=self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=[self.early_stop],
                       validation_data=(self.X_test, self.y_test),
                       verbose=self.verbose,
                       use_multiprocessing=self.multiprocess)
        return self.model

    def make_prediction(self, data=None):
        """make prediction on data- defaults to predict X_test"""
        if data is None:
            data = self.dataobject.X_test
        try:
            prediction = self.model.predict(self.scaler.transform(data))
            return prediction
        except ValueError:
            print('input data wrong shape-')
            print(f'correct_shape {self.dataobject.X_test.shape[-1]}')

    @staticmethod
    def choose_value(value_name: str,
                     choices=None, default=None) -> str | bool:
        """method to choose a value from a list.
        value_name: name of value
        choices: list of possible choices- str values
        default: what the default return is if
        no or not valid input.
        returns str of input or str of default.
        returns False if input not valid and no default
        is specified
        """
        if choices is None and default is None:
            return
        print(f'\nEnter desired {value_name}')
        print(f'Default is: {"Not active" if default is None else default}.')
        print(f'options: {"unspecified" if choices is None else choices}\n')
        choice = input(':')
        if choices is not None:
            if choice in choices:
                return choice
            elif default is not None:
                return str(default)
            print('choice not valid\n')
            return False
        if choice == '':
            return str(default)
        return choice

    def return_class_report(self):
        '''
        plots classifiers classification report
        '''
        from sklearn.metrics import classification_report
        if self.loss == 'rmse':
            print('This method is only accessible to classifiers')
            return
        elif self.loss == 'categorical_crossentropy':

            predicted = np.argmax(self.model.predict(
                self.X_test), axis=1)
            cls_df = pd.DataFrame(0, index=range(len(predicted)),
                                  columns=[x for x in range(
                                      0, self.y_train.shape[1])])
            for index, value in enumerate(predicted):
                cls_df.loc[index, value] = 1
            prediction = cls_df.values
        else:
            prediction = (self.model.predict(self.X_test) > 0.5)
        return classification_report(y_true=self.y_test, y_pred=prediction)

    def return_confusion_matrix(self):
        '''
        plots classifiers classification report
        '''
        from sklearn.metrics import confusion_matrix
        if self.loss == 'rmse':
            print('This method is only accessible to classifiers')
            return
        elif self.loss == 'categorical_crossentropy':
            predicted = np.argmax(self.model.predict(
                self.X_test), axis=1)
            cls_df = pd.DataFrame(0, index=range(len(predicted)),
                                  columns=[x for x in range(
                                      0, self.y_train.shape[1])])
            for index, value in enumerate(predicted):
                cls_df.loc[index, value] = 1
            prediction = cls_df.values
            return confusion_matrix(y_true=self.y_test.argmax(axis=1),
                                    y_pred=prediction.argmax(axis=1))
        else:
            prediction = (self.model.predict(self.X_test) > 0.5)
            return confusion_matrix(y_true=self.y_test, y_pred=prediction)

    def evaluate_regressor(self) -> list:
        """return r2, MAE and RMSE"""
        from sklearn.metrics import mean_absolute_error, \
            mean_squared_error, r2_score
        if self.loss != 'mse':
            print('This method is only accessible to regressors')
            return
        predictions = self.model.predict(self.X_test)
        RMSE = mean_squared_error(self.y_test, predictions)**0.5
        MAE = mean_absolute_error(self.y_test, predictions)
        r2_score = r2_score(self.y_test, predictions)
        return [MAE, RMSE, r2_score]

    def save_model(self, model_name: str):
        """save ANN model"""
        'from tensorflow.keras.models import save_model'
        model = self.model
        model.save(f'{model_name}.h5')
        print(f'model {model_name}.h5 is saved')
        return f'{model_name}.h5'


class Regressors:
    from sklearn.pipeline import Pipeline as Pipeline
    from sklearn.model_selection import GridSearchCV as GridSearchCV
    from sklearn.preprocessing import PolynomialFeatures as PolynomialFeatures
    from sklearn.preprocessing import StandardScaler as StandardScaler
    from sklearn.linear_model import LinearRegression as LinearRegression
    from sklearn.linear_model import Lasso as Lasso
    from sklearn.linear_model import Ridge as Ridge
    from sklearn.linear_model import ElasticNet as ElasticNet
    from sklearn.svm import SVR as SVR
    from sklearn.metrics import r2_score as r2_score

    def __init__(self, dataobject: PrepareData):
        self.dataobject = dataobject
        self.linear_tuple = self.create_linear(dataobject)
        self.lasso_tuple = self.create_lasso(dataobject)
        self.ridge_tuple = self.create_ridge(dataobject)
        self.elastic_tuple = self.create_elastic(dataobject)
        self.svr_tuple = self.create_svr(dataobject)
        self.ann_tuple = self.create_ann(dataobject)
        self.models = [self.linear_tuple, self.lasso_tuple, self.ridge_tuple,
                       self.elastic_tuple, self.svr_tuple, self.ann_tuple]

    def get_pipeline_grid(self, regression_type, param_grid: dict):
        """create a pipeline for model calling method"""
        pipeline = self.Pipeline(steps=[('scale', self.StandardScaler()),
                                        ('poly', self.PolynomialFeatures()),
                                        ('regression', regression_type())])
        grid_pipeline = self.GridSearchCV(pipeline, param_grid=param_grid,
                                          cv=10, scoring='r2')
        return grid_pipeline

    def create_linear(self, dataobject: PrepareData) -> Pipeline:
        """Create a linear regression"""
        self._param_grid_linear = {'poly__degree':
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        linear_pipe_grid = self.get_pipeline_grid(self.LinearRegression,
                                                  self._param_grid_linear)
        linear_pipe_grid.fit(dataobject.X_train, dataobject.y_train)
        best_params_linear = \
            linear_pipe_grid.best_params_['poly__degree']
        test_linear = linear_pipe_grid.best_estimator_
        test_linear.fit(dataobject.X_train, dataobject.y_train)
        self.linear_model = test_linear
        self.linear_params = {'poly': best_params_linear}
        return (self.linear_model, self.linear_params)

    def create_lasso(self, dataobject: PrepareData) -> Pipeline:
        """create lasso regression"""
        param_grid_lasso = {'poly__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            'regression__alpha': [0.1, 0.2, 0.30, 0.40, 0.5,
                                                  0.6, 0.7, 1, 5, 10, 50, 100],
                            'regression__tol': [0.3]}
        lasso_pipe_grid = self.get_pipeline_grid(self.Lasso, param_grid_lasso)
        lasso_pipe_grid.fit(dataobject.X_train, dataobject.y_train)
        poly_str = lasso_pipe_grid.best_params_['poly__degree']
        alpha_str = lasso_pipe_grid.best_params_['regression__alpha']
        tol_str = lasso_pipe_grid.best_params_['regression__tol']
        lasso_test = lasso_pipe_grid.best_estimator_
        lasso_test.fit(dataobject.X_train, dataobject.y_train)
        self.lasso_model = lasso_test
        self.lasso_params = {'poly': poly_str, 'alpha': alpha_str,
                             'regression_tol': tol_str}
        return (self.lasso_model, self.lasso_params)

    def create_ridge(self, dataobject: PrepareData) -> Pipeline:
        """create ridge regression"""
        ridge_param_grid = {'poly__degree': [1, 2, 3, 4, 5, 6, 7],
                            'regression__alpha': [0.1, 0.5, 1, 5, 10, 50, 100],
                            'regression__tol': [0.5]}
        ridge_pipe_grid = self.get_pipeline_grid(self.Ridge, ridge_param_grid)
        ridge_pipe_grid.fit(dataobject.X_train, dataobject.y_train)
        poly_str = ridge_pipe_grid.best_params_['poly__degree']
        alpha_str = ridge_pipe_grid.best_params_['regression__alpha']
        ridge_test = ridge_pipe_grid.best_estimator_
        ridge_test.fit(dataobject.X_train, dataobject.y_train)
        self.ridge_model = ridge_test
        self.ridge_params = {'poly': poly_str, 'alpha': alpha_str}
        return (self.ridge_model, self.ridge_params)

    def create_elastic(self, dataobject: PrepareData) -> Pipeline:
        """create elastic regression"""
        elastic_param_grid = {'poly__degree': [1, 2, 3, 4, 5, 6, 7],
                              'regression__alpha':
                              [0.1, 0.5, 1, 5, 10, 50, 100],
                              'regression__l1_ratio':
                              [.05, .1, .15, .2, .3, .5, .7, .9, .95, .99, 1],
                              'regression__tol': [0.5]}
        elastic_pipe_grid = self.get_pipeline_grid(self.ElasticNet,
                                                   elastic_param_grid)
        elastic_pipe_grid.fit(dataobject.X_train, dataobject.y_train)
        poly_str = elastic_pipe_grid.best_params_['poly__degree']
        alpha_str = elastic_pipe_grid.best_params_['regression__alpha']
        llratio_str = elastic_pipe_grid.best_params_['regression__l1_ratio']
        elastic_test = elastic_pipe_grid.best_estimator_
        elastic_test.fit(dataobject.X_train, dataobject.y_train)
        self.elastic_model = elastic_test
        self.elastic_params = {'poly': poly_str, 'alpha': alpha_str,
                               'l1ratio': llratio_str}
        return (self.elastic_model, self.elastic_params)

    def create_svr(self, dataobject: PrepareData) -> Pipeline:
        """create support vector machine regressor"""
        svr_param_grid = {'regression__C': [0.001, 0.01, 0.1, 0.5, 5, 10, 100],
                          'regression__kernel': ['linear', 'poly', 'rbf'],
                          'regression__degree': [2, 3, 4, 5, 6],
                          'regression__epsilon': [0, 0.01, 0.1, 0.5, 1, 2],
                          'regression__gamma': ['scale', 'auto']}
        svr_pipe_grid = self.get_pipeline_grid(self.SVR, svr_param_grid)
        svr_pipe_grid.fit(dataobject.X_train, dataobject.y_train)
        svr_test = svr_pipe_grid.best_estimator_
        svr_test.fit(dataobject.X_train, dataobject.y_train)
        c_str = svr_pipe_grid.best_params_['regression__C']
        kern_str = svr_pipe_grid.best_params_['regression__kernel']
        regdeg_str = svr_pipe_grid.best_params_['regression__degree']
        regeps_str = svr_pipe_grid.best_params_['regression__epsilon']
        regamm_str = svr_pipe_grid.best_params_['regression__gamma']
        self.svr_model = svr_test
        self.svr_params = {'C': c_str, 'kernel': kern_str,
                           'degree': regdeg_str, 'epsilon': regeps_str,
                           'gamma': regamm_str}
        return (self.svr_model, self.svr_params)

    def create_ann(self, dataobject: PrepareData) -> CreateANN:
        """method to create ANN model"""
        ann_model = CreateANN(dataobject, 'mse')
        if isinstance(ann_model.model, tf.keras.models.Sequential):
            ann_model.train_model()
            self.ann_model = ann_model
        else:
            print('model creation failed- no ann model created')
            self.ann_model = False
        return (self.ann_model, {'ann': 'no crossvalidation'})

    def score_models(self) -> list:
        """display the created models evaluate and save"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, \
            r2_score
        models_scored = []
        for model in self.models:
            if model is False:
                next
            if isinstance(model[0], CreateANN):
                ann_scores = model[0].evaluate_regressor()
                MAE = ann_scores[0]
                RMSE = ann_scores[1]
                r2 = ann_scores[2]
                model_name = 'ANN model'
            else:
                y_pred = model[0].predict(self.dataobject.X_test)
                model_name = model[0].steps[-1][1].__class__.__name__
                MAE = mean_absolute_error(y_true=self.dataobject.y_test,
                                          y_pred=y_pred)
                RMSE = np.sqrt(mean_squared_error(
                    y_true=self.dataobject.y_test, y_pred=y_pred))
                r2 = r2_score(y_true=self.dataobject.y_test, y_pred=y_pred)
            model_dict = {'name': model_name, 'MAE': MAE,
                          'RMSE': RMSE, 'R2_score': r2,
                          'best_params': model[1],
                          'model': model[0]}
            models_scored.append(model_dict)
        self.models_scored = models_scored
        return self.models_scored

    def display_model_evaluation(self):
        """display model metrics and select model to save"""
        best_r2 = 0
        best_model_name = ''
        best_model = ''
        for model in self.models_scored:
            print(f'\n\nModel: {model["name"]}')
            print(f'MAE: {model["MAE"]}')
            print(f'RMSE: {model["RMSE"]}')
            print(f'r2_score: {model["R2_score"]}')
            print(f'best params: {model["best_params"]}')
            if model['R2_score'] > best_r2:
                best_r2 = model['R2_score']
                best_model = model['model']
                best_model_name = model['name']
        print(f'\n\nSuggested model: {best_model_name}')
        print(f'with an r2 score of {best_r2}')
        model_save = input(
            f'enter name of model to save (default:{best_model_name})')
        saved_model = ''
        for model in self.models_scored:
            if str(model_save) == str(model['name']):
                saved_model = model['model']
        if saved_model == '':
            print('no model chosen or found- saves suggested model')
            saved_model = best_model
        return saved_model


class Classifiers:
    from sklearn.pipeline import Pipeline as Pipeline
    from sklearn.model_selection import GridSearchCV as GridSearchCV
    from sklearn.preprocessing import StandardScaler as StandardScaler
    from sklearn.linear_model import LogisticRegression as LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
    from sklearn.svm import SVC as SVC
    from sklearn.metrics import accuracy_score as accuracy_score

    def __init__(self, dataobject: PrepareData):
        self.dataobject = dataobject
        dataobject.split_train_test(X=dataobject.X_dummies,
                                    y=dataobject.y, test_size=0.30,
                                    random_state=101)
        self.logistic_tuple = self.create_logistic(dataobject)
        self.knn_tuple = self.create_knn(dataobject)
        self.svc_tuple = self.create_svc(dataobject)
        self.ann_tuple = self.create_ann(dataobject)
        self.models = [self.logistic_tuple, self.knn_tuple,
                       self.svc_tuple, self.ann_tuple]

    def get_pipeline_grid(self, model, param_grid: dict):
        """create a pipeline + grid search cv for model and return it"""
        pipeline = self.Pipeline([('scaler', self.StandardScaler()),
                                  ('model', model())])
        pipe_grid = self.GridSearchCV(estimator=pipeline,
                                      param_grid=param_grid,
                                      cv=10, scoring='accuracy')
        return pipe_grid

    def create_logistic(self, dataobject: PrepareData):
        """create logistic classifier"""
        logistic_param_grid = {'model__C': [0.001, 0.1, 10, 100, 10e5],
                               'model__penalty': ['l2']}
        logistic_pipe_grid = self.get_pipeline_grid(self.LogisticRegression,
                                                    logistic_param_grid)
        logistic_pipe_grid.fit(self.dataobject.X_train,
                               self.dataobject.y_train)
        logistic_test = logistic_pipe_grid.best_estimator_
        logistic_test.fit(self.dataobject.X_train, self.dataobject.y_train)
        c_str = logistic_pipe_grid.best_params_['model__C']
        penalty_str = logistic_pipe_grid.best_params_['model__penalty']
        self.logistic_model = logistic_test
        self.logistic_params = {'c': c_str, 'penalty': penalty_str}
        return (self.logistic_model, self.logistic_params)

    def create_knn(self, dataobject: PrepareData):
        """create key nearest neighbor classifier"""
        knn_param_grid = {'model__n_neighbors': list(range(1, 31)), }
        knn_pipe_grid = self.get_pipeline_grid(self.KNeighborsClassifier,
                                               knn_param_grid)
        knn_pipe_grid.fit(dataobject.X_train, dataobject.y_train)
        n_str = knn_pipe_grid.best_params_['model__n_neighbors']
        knn_test = knn_pipe_grid.best_estimator_
        knn_test.fit(dataobject.X_train, dataobject.y_train)
        self.knn_model = knn_test
        self.knn_params = {'neighbors': n_str}
        return (self.knn_model, self.knn_params)

    def create_svc(self, dataobject: PrepareData):
        """create support vector classifier"""
        svc_param_grid = {'model__C': [0.001, 0.1, 10, 100],
                          'model__kernel':
                          ['linear', 'poly', 'sigmoid', 'rbf'],
                          'model__degree': [2, 3, 4, 5, 6],
                          'model__gamma': [0.1, 0.01]}
        svc_pipe_grid = self.get_pipeline_grid(self.SVC, svc_param_grid)
        svc_pipe_grid.fit(dataobject.X_train, dataobject.y_train)
        c_str = svc_pipe_grid.best_params_['model__C']
        kernel_str = svc_pipe_grid.best_params_['model__kernel']
        degree_str = svc_pipe_grid.best_params_['model__degree']
        gamma_str = svc_pipe_grid.best_params_['model__gamma']
        svc_test = svc_pipe_grid.best_estimator_
        svc_test.fit(dataobject.X_train, dataobject.y_train)
        self.svc_model = svc_test
        self.svc_params = {'C': c_str, 'kernel': kernel_str,
                           'degree': degree_str, 'gamma': gamma_str}
        return (self.svc_model, self.svc_params)

    def create_ann(self, dataobject):
        """call ANN class and return ann object"""
        if dataobject.y.nunique() > 2:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'
        ann_model = CreateANN(dataobject, loss)
        if isinstance(ann_model.model, tf.keras.models.Sequential):
            ann_model.train_model()
            self.ann_model = ann_model
        else:
            print('model creation failed- no ann model created')
            self.ann_model = False
        return (self.ann_model, {'Ann': 'no grid search'})

    def score_models(self):
        """ model evaluation"""
        dataobject = self.dataobject
        from sklearn.metrics import classification_report, confusion_matrix
        models_scored = []
        for model in self.models:
            if model[0] is False:
                next
            if isinstance(model[0], CreateANN):
                model_name = 'ANN model'
                cls_report = model[0].return_class_report()
                con_matrix = model[0].return_confusion_matrix()
            else:
                y_pred = model[0].predict(self.dataobject.X_test)
                model_name = model[0].steps[-1][1].__class__.__name__
                con_matrix = confusion_matrix(y_true=dataobject.y_test,
                                              y_pred=y_pred,
                                              labels=model[0].classes_)
                cls_report = classification_report(y_true=dataobject.y_test,
                                                   y_pred=y_pred)
            model_dict = {'name': model_name, 'confusion_matrix': con_matrix,
                          'classification_report': cls_report,
                          'best_params': model[1], 'model': model[0]}
            models_scored.append(model_dict)
            self.models_scored = models_scored
        return self.models_scored

    def display_model_evaluation(self):
        """display reports and suggest model by accuracy score"""
        import re
        from sklearn.metrics import ConfusionMatrixDisplay
        best_accuracy = 0
        best_model = ''
        best_model_name = ''
        for model_dict in self.models_scored:
            print(f'\nmodel: {model_dict["name"]}')
            print('Confusion matrix:')
            ConfusionMatrixDisplay(
                confusion_matrix=model_dict['confusion_matrix']).plot()
            print(
                f'classification report:\n\
                    {model_dict["classification_report"]}'
                )
            print(f'best_parameters:{model_dict["best_params"]}')
            accuracy = re.search(r'accuracy\s+(\d+\.\d+)',
                                 str(model_dict['classification_report']))
            if accuracy:
                accuracy = float(accuracy.group(1))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_dict['model']
                    best_model_name = model_dict['name']
        print(f'\n\nsuggested model is {best_model_name}')
        print(f'with an accuracy of {best_accuracy}\n\n')
        model_save = input(f'Enter model name else {best_model_name} is saved')
        saved_model = ''
        for model_dict in self.models_scored:
            if model_save == model_dict['name']:
                saved_model = model_dict['model']
                print(f'model {model_dict["name"]} selected')
        if saved_model == '':
            print('no model chosen or found- saves suggested model')
            saved_model = best_model
        return saved_model
