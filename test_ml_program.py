import unittest
from unittest import TestCase
import pandas as pd
import numpy as np
import tensorflow as tf
from unittest.mock import patch
from ml_program import PrepareData, CreateModel
from ml_program import Regressors, Classifiers, CreateANN
import os


TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TEST_CSV_DIRECTORY = os.path.join(TEST_DIRECTORY, 'test_data')

CATEGORICAL_NO_MISSING = os.path.join(TEST_CSV_DIRECTORY, 'penguins_nona.csv')
CATEGORICAL_MISSING = os.path.join(TEST_CSV_DIRECTORY, 'penguins_size.csv')
NUMERICAL_NO_MISSING = os.path.join(TEST_CSV_DIRECTORY, 'gene_expression.csv')
NUMERICAL_MISSING = os.path.join(TEST_CSV_DIRECTORY,
                                 'Advertising_missing.csv')
NON_EXISTING_DATA = 'i_do_not_exist.csv'
COLUMN_NAME_NUMERICAL = 'Gene One'
COLUMN_NAME_NON_EXISTING = 'i do not exist'  #
COLUMN_NAME_MULTICLASS = 'island'
COLUMN_NAME_BINARYCLASS = 'sex'
COLUMN_NUMERICAL_CAT_DATA = 'body_mass_g'
NUMERICAL_BINARY_COLUMN = 'Cancer Present'


if __name__ == '__main__':
    unittest.main()


class Test_CreateModel(TestCase):

    @patch('ml_program.Regressors')
    @patch('ml_program.CreateModel.choose_model')
    def test_constructor_regressor(self,
                                   mock_choose_model, mock_regressor_cls):
        """verify constructor create regressor"""
        csv_file = NUMERICAL_NO_MISSING
        mock_choose_model.return_value = 'reg'
        mock_regressor_cls.return_value = 'Regressor'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  'reg']):
            create_modlel_obj = CreateModel()
        self.assertIsInstance(create_modlel_obj, CreateModel)
        self.assertEqual(create_modlel_obj.model_type, 'Regressor')

    @patch('ml_program.Classifiers')
    @patch('ml_program.CreateModel.choose_model')
    def test_constructor_classifier(self,
                                    mock_choose_model, mock_classifier_cls):
        """verify contstructor create classifier"""
        csv_file = CATEGORICAL_NO_MISSING
        mock_choose_model.return_value = 'cat'
        mock_classifier_cls.return_value = 'Classifier'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  'cat']):
            create_modlel_obj = CreateModel()
        self.assertIsInstance(create_modlel_obj, CreateModel)
        self.assertTrue(create_modlel_obj.dataobj.print_why_not_ready())
        self.assertEqual(create_modlel_obj.model_type, 'Classifier')

    @patch('ml_program.CreateModel.choose_model')
    def test_constructor_no_model(self, mock_choose_model):
        """verify constructor creates no model if bad input"""
        csv_file = NUMERICAL_NO_MISSING
        mock_choose_model.return_value = False
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  'reg']):
            create_modlel_obj = CreateModel()
        self.assertIsInstance(create_modlel_obj, CreateModel)
        self.assertFalse(create_modlel_obj.model_type)

    def test_choose_model_correct(self):
        """Verify enter 1 or 2 in choose model returns str"""
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  'reg']):
            create_modlel_obj = CreateModel()
        self.assertIsInstance(create_modlel_obj, CreateModel)
        self.assertFalse(create_modlel_obj.model_cat)
        with patch('builtins.input', side_effect='1'):
            create_modlel_obj.choose_model()
        self.assertTrue(create_modlel_obj.model_cat)

    @patch('ml_program.Regressors')
    def test_choose_model_wrong(self, mock_regressor):
        """verify enter not 1 or 2 returns false"""
        csv_file = NUMERICAL_NO_MISSING
        mock_regressor.return_value = 'Regressor class'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  '1']):
            create_modlel_obj = CreateModel()
        self.assertIsInstance(create_modlel_obj, CreateModel)
        self.assertTrue(create_modlel_obj.model_cat)
        with patch('builtins.input', return_value='3'):
            create_modlel_obj.choose_model()
        self.assertFalse(create_modlel_obj.model_cat)

    @patch('ml_program.CreateANN.save_model')
    def test_save_model_keras(self, mock_save_keras):
        """verify ann save method is called when ann model choosen"""
        mock_save_keras.return_value = 'SAVED'
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'mse')
        ann_obj.train_model()
        with patch('builtins.input', return_value='test_model'):
            self.assertEqual(model_creator.save_model(ann_obj.model),
                             'SAVED')

    @patch('ml_program.Regressors.create_ann')
    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_linear')
    def test_save_model_sklearn(self, mock_lasso, mock_ridge,
                                mock_elastic, mock_svr, mock_ann):
        """verify sk_learn model is saved as .joblib file"""
        import os
        mock_lasso.return_value = 'Lasso Regression'
        mock_ridge.return_value = 'Ridge Regression'
        mock_elastic.return_value = 'Elastic Net Regression'
        mock_svr.return_value = 'SVR Model'
        mock_ann.return_value = 'Ann model'
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        regressor_object = Regressors(data_obj)
        linear = regressor_object.linear_tuple[0]
        with patch('builtins.input', return_value='test_linear'):
            model_creator.save_model(linear)
        self.assertTrue(os.path.isfile('test_linear.joblib'))


class Test_PrepareData(TestCase):

    def test_constructor(self):
        """verify construcor return PrepareData object"""
        with patch('builtins.input',
                   return_value=NUMERICAL_MISSING):
            test_prep_obj = PrepareData()
        self.assertIsInstance(test_prep_obj, PrepareData)

    def test_verify_csv(self):
        "confirm verify_csv return True if file found"
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        self.assertIsNotNone(test_prep_obj)
        self.assertIsInstance(test_prep_obj, PrepareData)

    def test_verify_csv_raise_error(self):
        """confirm verify_csv returns False if file not found"""
        data_path = NON_EXISTING_DATA
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        with self.assertRaises(FileNotFoundError):
            test_prep_obj.verify_csv()
            self.assertFalse(test_prep_obj)

    def test_confirm_no_na_true(self):
        """confirm_no_na method return True if no missing values"""
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        self.assertTrue(test_prep_obj.confirm_no_na())

    def test_confirm_no_na_false(self):
        """confirm_no_na method return False if missing values"""
        data_path = NUMERICAL_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        self.assertFalse(test_prep_obj.confirm_no_na())

    def test_choose_target_correct(self):
        """verify valid input returns name of target col"""
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_NUMERICAL):
            self.assertEqual(test_prep_obj.choose_target(),
                             COLUMN_NAME_NUMERICAL)

    def test_choose_target_fail(self):
        "verify not valid target input returns False"
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_NON_EXISTING):
            self.assertFalse(test_prep_obj.choose_target(), 'saes')

    def test_split_dependent_independent_correct(self):
        "confirm X and y are split properly"
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_NUMERICAL):
            test_prep_obj.choose_target()
        X, y = test_prep_obj.split_dependent_independent()
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(y), len(test_prep_obj.df))

    def test_split_dependent_independent_fail(self):
        "confirm x and y return False if nonexisting target y"
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input',
                   return_value=COLUMN_NAME_NON_EXISTING):
            test_prep_obj.choose_target()
        X, y = test_prep_obj.split_dependent_independent()
        self.assertFalse(X)
        self.assertFalse(y)
        self.assertIsInstance(test_prep_obj, PrepareData)

    def test_split_train_test_correct(self):
        "confirm train/test split return valid train/test x/y"
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_NUMERICAL):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        test_size = 0.30
        random_state = 101
        X_train, X_test, y_train, y_test = \
            test_prep_obj.split_train_test(X=test_prep_obj.X,
                                           y=test_prep_obj.y,
                                           test_size=test_size,
                                           random_state=random_state)
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertTrue(any([isinstance(y_test, pd.Series),
                             isinstance(y_test, pd.DataFrame)]))
        self.assertTrue(any([isinstance(y_train, pd.Series),
                             isinstance(y_train, pd.DataFrame)]))
        self.assertEqual(len(X_train)+len(X_test), len(test_prep_obj.df))
        self.assertEqual(len(y_train)+len(y_test), len(test_prep_obj.df))
        self.assertEqual(test_size, test_prep_obj.test_size)
        self.assertEqual(random_state, test_prep_obj.random_state)

    def test_split_train_test_fail(self):
        """X_train/test, y_train/test return false if bad random_state"""
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_NUMERICAL):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        test_size = 0.30
        random_state = 'X'
        X_train, X_test, y_train, y_test = \
            test_prep_obj.split_train_test(X=test_prep_obj.X,
                                           y=test_prep_obj.y,
                                           test_size=test_size,
                                           random_state=random_state)
        self.assertFalse(X_test)
        self.assertFalse(X_train)
        self.assertFalse(y_train)
        self.assertFalse(y_test)

    def test_get_y_dummies_reg_num(self):
        """return series if target need no dunmmy"""
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_NUMERICAL):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        self.assertIsInstance(test_prep_obj.get_y_dummies('reg'),
                              pd.Series)

    def test_get_y_dummies_cat_num_no(self):
        """return False if target is num and too varied for dummy"""
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_NUMERICAL):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        self.assertFalse(test_prep_obj.get_y_dummies('cat'),
                         pd.Series)

    def test_get_y_dummies_cat_num_yes(self):
        """return dummy df if target is num and can be converted"""
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=NUMERICAL_BINARY_COLUMN):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        self.assertIsInstance(test_prep_obj.get_y_dummies('cat'),
                              pd.DataFrame)

    def test_get_y_dummies_cat_clas(self):
        """return dummies for classifier choice w object data"""
        data_path = CATEGORICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_MULTICLASS):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        self.assertIsInstance(test_prep_obj.get_y_dummies('cat'),
                              pd.DataFrame)

    def test_get_y_dummies_cat_reg(self):
        """return False for regressor choice w object data"""
        data_path = CATEGORICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_MULTICLASS):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        self.assertFalse(test_prep_obj.get_y_dummies('reg'))

    def test_get_x_dummies(self):
        """confirm dummies are created"""
        data_path = CATEGORICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=COLUMN_NAME_MULTICLASS):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        test_prep_obj.get_x_dummies()
        self.assertIsInstance(test_prep_obj.X_dummies, pd.DataFrame)
        self.assertNotEqual(test_prep_obj.X_dummies.shape,
                            test_prep_obj.df.drop('island', axis=1).shape)

    def test_get_x_dummies_no_dummies(self):
        """confirm X df has the same shape after dummy method"""
        data_path = NUMERICAL_NO_MISSING
        with patch('builtins.input', return_value=data_path):
            test_prep_obj = PrepareData()
        test_prep_obj.verify_csv()
        with patch('builtins.input', return_value=NUMERICAL_BINARY_COLUMN):
            test_prep_obj.choose_target()
        test_prep_obj.split_dependent_independent()
        test_prep_obj.get_x_dummies()
        self.assertIsInstance(test_prep_obj.X_dummies, pd.DataFrame)
        self.assertEqual(test_prep_obj.X_dummies.shape,
                         test_prep_obj.df.drop(NUMERICAL_BINARY_COLUMN,
                                               axis=1).shape)

    def test_print_why_not_ready_ready(self):
        """confirm that method return True if no problem with data"""
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        self.assertTrue(all([vars(data_obj).values]))
        self.assertTrue(data_obj.print_why_not_ready())

    def test_print_why_not_ready_not_ready(self):
        """confirm method return False if problems any in data"""
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
                self.assertTrue(data_obj.print_why_not_ready())
        with patch('builtins.input', return_value='sls'):
            self.assertFalse(data_obj.choose_target())
            self.assertFalse(data_obj.target)
        self.assertFalse(data_obj.print_why_not_ready())


class Test_CreateANN(TestCase):

    def test_constructor(self):
        """confirm constructor takes valid input and creates model"""
        import tensorflow as tf
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NUMERICAL_CAT_DATA,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj

        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'mse')
        self.assertIsInstance(ann_obj, CreateANN)
        self.assertTrue(ann_obj.loss == 'mse')
        self.assertIsInstance(ann_obj.model, tf.keras.models.Sequential)

    def test_get_data_regressioin(self):
        """test that df if accessed from dataobject
        test that y is only one value
        """
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NUMERICAL_CAT_DATA,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'mse')
        X, y = ann_obj.get_data(data_obj)
        print(y)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.shape, (len(y),))

    def test_get_data_binary(self):
        """test that df from dataobject is returned for multiclass
        test that y is more than one column
        """
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_BINARYCLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'binary_crossentropy')
        X, y = ann_obj.get_data(data_obj)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(y.shape), 1)
        self.assertEqual(y.shape, (len(y), ))

    def test_get_data_multiclass(self):
        """test that df if accessed from dataobject"""
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        X, y = ann_obj.get_data(data_obj)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(y.shape[1] > 1)

    def test_split_data(self):
        """split train/test"""
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        X, y = ann_obj.get_data(data_obj)
        X_train, X_test, y_train, y_test = ann_obj.split_data(X, y, data_obj)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(sum([len(X_test), len(X_train)]), len(X))
        self.assertEqual(sum([len(y_test), len(y_train)]), len(y))
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertIsInstance(X_train, np.ndarray)

    def test_scale_data(self):
        from sklearn.model_selection import train_test_split
        import copy
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        X, y = ann_obj.get_data(data_obj)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=101,
                                                            test_size=0.33)
        scaler_1 = copy.deepcopy(ann_obj.scaler)
        X_train_scaled, X_test_scaled = ann_obj.scale_data(X_train, X_test)
        self.assertEqual(len(X_train), len(X_train_scaled))
        self.assertEqual(len(X_test), len(X_test_scaled))
        self.assertFalse(scaler_1 == ann_obj.scaler)
        self.assertFalse(np.array_equal(X_train, X_train_scaled))
        self.assertFalse(np.array_equal(X_test, X_test_scaled))

    @patch('ml_program.CreateANN.build_model')
    def test_define_hidden_layer_tuple(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(layer_tuple, ann_obj.layer_tuple)
                self.assertTrue(ann_obj.layer_tuple, tuple)
        with patch('builtins.input',
                   side_effect=['11', '-0.33', '-3', 'x']):
            new_tuple = ann_obj.define_hidden_layer_tuple()
            self.assertIsInstance(new_tuple, tuple)
            self.assertEqual(len(new_tuple), 2)
            self.assertTrue(new_tuple == (11, -0.33))
        with patch('builtins.input', side_effect=['x', '33', '-3', 'x']):
            new_tuple = ann_obj.define_hidden_layer_tuple()
            self.assertIsInstance(new_tuple, tuple)
            self.assertEqual(new_tuple, (100,))

    @patch('ml_program.CreateANN.build_model')
    def test_choose_activation(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.activation, activation)
        with patch('builtins.input', return_value='tanh'):
            new_activation = ann_obj.choose_activation()
            self.assertEqual(new_activation, 'tanh')
        with patch('builtins.input', return_value='tah'):
            new_activation = ann_obj.choose_activation()
            self.assertEqual(new_activation, 'relu')
        self.assertIsInstance(ann_obj.activation, str)

    @patch('ml_program.CreateANN.build_model')
    def test_choose_optimizer(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.optimizer, optimizer)
        with patch('builtins.input', return_value='sgd'):
            new_optimizer = ann_obj.choose_optimizer()
            self.assertEqual(new_optimizer, 'sgd')
        with patch('builtins.input', return_value='tah'):
            new_optimizer = ann_obj.choose_optimizer()
            self.assertEqual(new_optimizer, 'adam')
        self.assertIsInstance(ann_obj.optimizer, str)

    @patch('ml_program.CreateANN.build_model')
    def test_choose_batch_size(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.batch_size, int(batch_size))
        with patch('builtins.input', return_value='22'):
            new_batch = ann_obj.choose_batch_size()
            self.assertEqual(new_batch, 22)
        with patch('builtins.input', return_value='not a number'):
            new_batch = ann_obj.choose_batch_size()
            self.assertEqual(new_batch, 32)
        self.assertIsInstance(ann_obj.batch_size, int)

    @patch('ml_program.CreateANN.build_model')
    def test_choose_epochs(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.epochs, int(epochs))
        with patch('builtins.input', return_value='22'):
            new_epochs = ann_obj.choose_epochs()
            self.assertEqual(new_epochs, 22)
        with patch('builtins.input', return_value='not a number'):
            new_epochs = ann_obj.choose_epochs()
            self.assertEqual(new_epochs, 1)
        self.assertIsInstance(ann_obj.batch_size, int)

    @patch('ml_program.CreateANN.build_model')
    def test_choose_monitor(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'val_loss'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.monitor, monitor)
        with patch('ml_program.CreateANN.choose_value',
                   return_value='accuracy'):
            self.assertEqual(ann_obj.choose_monitor(), 'accuracy')
        with patch('ml_program.CreateANN.choose_value',
                   return_value='val_loss'):
            self.assertEqual(ann_obj.choose_monitor(), 'val_loss')

    @patch('ml_program.CreateANN.build_model')
    def test_choose_patience(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.patience, int(patience))
        with patch('builtins.input', return_value='22'):
            new_patience = ann_obj.choose_patience()
            self.assertEqual(new_patience, 22)
        with patch('builtins.input', return_value='not a number'):
            new_patience = ann_obj.choose_patience()
            self.assertEqual(new_patience, ann_obj.epochs)
        self.assertIsInstance(ann_obj.batch_size, int)

    @patch('ml_program.CreateANN.build_model')
    def test_choose_mode(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.mode, mode)
        with patch('ml_program.CreateANN.choose_value', return_value='max'):
            self.assertTrue(ann_obj.choose_mode() == 'max')

    @patch('ml_program.CreateANN.build_model')
    def test_choose_verbose(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertEqual(ann_obj.verbose, int(verbose))
        with patch('builtins.input', return_value='2'):
            new_verbose = ann_obj.choose_verbose()
            self.assertEqual(new_verbose, 2)
        with patch('builtins.input', return_value='77'):
            new_verbose = ann_obj.choose_verbose()
            self.assertFalse(new_verbose)

    @patch('ml_program.CreateANN.build_model')
    def test_choose_multiprocess(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
                self.assertFalse(ann_obj.multiprocess)
        with patch('ml_program.CreateANN.choose_value', return_value='True'):
            self.assertTrue(ann_obj.choose_multiprocess())
        with patch('ml_program.CreateANN.choose_value', return_value='Hello'):
            self.assertFalse(ann_obj.choose_multiprocess())

    @patch('ml_program.CreateANN.build_model')
    def test_get_output_activation(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        self.assertEqual(ann_obj.get_output_activation(), 'softmax')

    @patch('ml_program.CreateANN.build_model')
    def test_get_output_activation_fail(self, mock_model_build):
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'not a valid loss name')
        self.assertFalse(ann_obj.get_output_activation())

    def test_build_model(self):
        import tensorflow as tf
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                with patch('ml_program.CreateANN.build_model',
                           return_value='ANN'):
                    ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        with patch('ml_program.CreateANN.get_output_activation',
                   return_valu='softmax'):
            test_model_build = ann_obj.build_model()
            self.assertIsInstance(test_model_build, tf.keras.models.Sequential)

    def test_build_model_false(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                with patch('ml_program.CreateANN.build_model',
                           return_value='ANN'):
                    ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        with patch('ml_program.CreateANN.get_output_activation',
                   return_value=False):
            test_model_build = ann_obj.build_model()
            self.assertFalse(test_model_build)

    @patch('ml_program.CreateANN.build_model')
    def test_add_hidden_layers(self, mock_model_build):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        csv_file = CATEGORICAL_NO_MISSING
        mock_model_build.return_value = 'ANN Build'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        test_model = Sequential()
        test_model.add(Dense(units=9, activation=activation))
        test_model = ann_obj.add_hidden_layers(test_model)
        expected_layers = [
            {'type': 'Dense', 'units': 9, 'activation': 'relu'},
            {'type': 'Dense', 'units': 2, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.22, 'seed': 101}
            ]
        for i, layer in enumerate(test_model.layers):
            expected_config = expected_layers[i]
            self.assertEqual(layer.__class__.__name__, expected_config['type'])

    def test_train_model_regressor(self):
        import tensorflow as tf
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NUMERICAL_CAT_DATA,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regrressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'mse')
        self.assertIsInstance(ann_obj.model, tf.keras.models.Sequential)
        ann_trained = ann_obj.train_model()
        self.assertIsInstance(ann_trained, tf.keras.models.Sequential)

    def test_train_model_multiclass(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        self.assertIsInstance(ann_obj.model, tf.keras.models.Sequential)
        ann_trained = ann_obj.train_model()
        self.assertIsInstance(ann_trained, tf.keras.models.Sequential)

    def test_train_model_binary(self):
        import tensorflow as tf
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_BINARYCLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'binary_crossentropy')
        self.assertIsInstance(ann_obj.model, tf.keras.models.Sequential)
        self.assertEqual(ann_obj.loss, 'binary_crossentropy')
        self.assertEqual('sigmoid', ann_obj.get_output_activation())
        ann_trained = ann_obj.train_model()
        self.assertIsInstance(ann_trained, tf.keras.models.Sequential)

    def test_choose_value(self):
        """assure valid input return valid output"""
        choices = ['11', '2', '3']
        default = None
        with patch('builtins.input', return_value='11'):
            choice = CreateANN.choose_value(value_name='Test method',
                                            choices=choices, default=default)
            self.assertIsInstance(choice, str)
            self.assertEqual(choice, '11')

    def test_choose_value_false(self):
        """assure not valid input and no default return False"""
        choices = ['11', '2', '3']
        default = None
        with patch('builtins.input', return_value='1'):
            choice = CreateANN.choose_value(value_name='Test method',
                                            choices=choices, default=default)
        self.assertFalse(choice)

    def test_choose_value_default(self):
        """assure not valid input return default as str"""
        choices = ['11', '2', '3']
        default = 3
        with patch('builtins.input', return_value='1'):
            choice = CreateANN.choose_value(value_name='Test method',
                                            choices=choices, default=default)
        self.assertIsInstance(choice, str)
        self.assertEqual(choice, '3')

    def test_return_class_report_binary(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_BINARYCLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'binary_crossentropy')
        ann_obj.train_model()
        self.assertIsInstance(ann_obj.return_class_report(), str)

    def test_return_class_report_multiclass(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        ann_obj.train_model()
        self.assertIsInstance(ann_obj.return_class_report(), str)

    def test_return_confusion_matxix_binary(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_BINARYCLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'binary_crossentropy')
        ann_obj.train_model()
        self.assertIsInstance(ann_obj.return_confusion_matrix(), np.ndarray)

    def test_return_confusion_matrix_multiclass(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'categorical_crossentropy')
        ann_obj.train_model()
        self.assertIsInstance(ann_obj.return_confusion_matrix(), np.ndarray)

    def test_evaluate_regressor(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NUMERICAL_CAT_DATA,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'mse')
        ann_obj.train_model()
        evaluations = ann_obj.evaluate_regressor()
        self.assertIsInstance(evaluations, list)
        self.assertTrue(len(evaluations) == 3)
        for evaluation in evaluations:
            self.assertIsInstance(evaluation, (int, float))

    def test_save_model(self):
        """assert model is saved"""
        import os
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NUMERICAL_CAT_DATA,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, -0.3, 2)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_obj = CreateANN(data_obj, 'mse')
        ann_obj.train_model()
        model_name = 'test_Ann'
        ann_obj.save_model(model_name)
        self.assertTrue(os.path.isfile('test_Ann.h5'))


class Test_Regressors(TestCase):
    from sklearn.linear_model import LinearRegression as LinearRegression
    from sklearn.linear_model import Lasso as Lasso
    from sklearn.linear_model import Ridge as Ridge
    from sklearn.linear_model import ElasticNet as ElasticNet
    from sklearn.svm import SVR as SVR
    from sklearn.model_selection import GridSearchCV as GridSearchCV
    from sklearn.pipeline import Pipeline as Pipeline

    @patch('ml_program.Regressors.create_ann')
    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_lasso')
    @patch('ml_program.Regressors.create_linear')
    @patch('ml_program.PrepareData')
    def test_get_pipeline_grid(self, mock_data, mock_linear,
                               mock_lasso, mock_ridge, mock_elastic,
                               mock_svr, mock_ann):
        """make sure get_pipeline_grid return grid/pipe obj"""
        mock_data.return_value = 'Data'
        mock_linear.return_value = 'Linear'
        mock_lasso.return_value = 'Lasso'
        mock_ridge.return_value = 'Ridge'
        mock_elastic.return_value = 'Elastic'
        mock_svr.return_value = 'SVR'
        mock_ann.return_value = 'ANN'
        regressor_object = Regressors(PrepareData)
        param_grid = {'poly__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        self.assertIsInstance(
            regressor_object.get_pipeline_grid(self.LinearRegression,
                                               param_grid),
            self.GridSearchCV)

    @patch('ml_program.Regressors.create_ann')
    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_lasso')
    def test_create_linear(self, mock_lasso, mock_ridge,
                           mock_elastic, mock_svr, mock_ann):
        """make sure create linear return regressor object"""
        mock_lasso.return_value = 'Lasso Regression'
        mock_ridge.return_value = 'Ridge Regression'
        mock_elastic.return_value = 'Elastic Net Regression'
        mock_svr.return_value = 'SVR Model'
        mock_ann.return_value = 'Ann model'
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Regressors.create_linear',
                   return_value='False'):
            regressor_object = Regressors(data_obj)
        linear_tuple = regressor_object.create_linear(data_obj)
        linear_mod = linear_tuple[0]
        linear_params = linear_tuple[1]
        self.assertIsInstance(linear_tuple, tuple)
        self.assertIsInstance(linear_params, dict)
        self.assertIsInstance(linear_mod, self.Pipeline)
        self.assertIsInstance(linear_mod.steps[-1][1], self.LinearRegression)

    @patch('ml_program.Regressors.create_ann')
    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_linear')
    def test_create_lasso(self, mock_linear, mock_ridge,
                          mock_elastic, mock_svr, mock_ann):
        """make sure create_lasso returns lasso regressor/pipeline"""
        mock_linear.return_value = 'Linear Regression'
        mock_ridge.return_value = 'Ridge regression'
        mock_elastic.return_value = 'Elastic Net Regression'
        mock_svr.return_value = 'SVR Model'
        mock_ann.return_value = 'Ann model'
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Regressors.create_lasso',
                   return_value='False'):
            regressor_object = Regressors(data_obj)
        lasso_tuple = regressor_object.create_lasso(data_obj)
        lasso_mod = lasso_tuple[0]
        lasso_params = lasso_tuple[1]
        self.assertIsInstance(lasso_tuple, tuple)
        self.assertIsInstance(lasso_params, dict)
        self.assertIsInstance(lasso_mod, self.Pipeline)
        self.assertIsInstance(lasso_mod.steps[-1][1], self.Lasso)

    @patch('ml_program.Regressors.create_ann')
    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_lasso')
    @patch('ml_program.Regressors.create_linear')
    def test_create_ridge(self, mock_linear, mock_lasso,
                          mock_elastic, mock_svr, mock_ann):
        """Make sure create_ridge returns pipeline with Ridge"""
        mock_linear.return_value = 'Linear Regression'
        mock_lasso.return_value = 'Lasso Regression'
        mock_elastic.return_value = 'Elastic Net Regression'
        mock_svr.return_value = 'SVR Model'
        mock_ann.return_value = 'Ann model'
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Regressors.create_ridge',
                   return_value='False'):
            regressor_object = Regressors(data_obj)
        ridge_tuple = regressor_object.create_ridge(data_obj)
        ridge_mod = ridge_tuple[0]
        ridge_params = ridge_tuple[1]
        self.assertIsInstance(ridge_tuple, tuple)
        self.assertIsInstance(ridge_params, dict)
        self.assertIsInstance(ridge_mod, self.Pipeline)
        self.assertIsInstance(ridge_mod.steps[-1][1], self.Ridge)

    @patch('ml_program.Regressors.create_ann')
    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_lasso')
    @patch('ml_program.Regressors.create_linear')
    def test_create_elastic(self, mock_linear, mock_lasso,
                            mock_ridge, mock_svr, mock_ann):
        """make sure elastic net method return pipeline object w ElasticNet"""
        mock_linear.return_value = 'Linear Regression'
        mock_lasso.return_value = 'Lasso Regression'
        mock_ridge.return_value = 'Ridge Regression'
        mock_svr.return_value = 'SVR model'
        mock_ann.return_value = 'Ann model'
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Regressors.create_elastic',
                   return_value='False'):
            regressor_object = Regressors(data_obj)
        elastic_tuple = regressor_object.create_elastic(data_obj)
        elastic_mod = elastic_tuple[0]
        elastic_params = elastic_tuple[1]
        self.assertIsInstance(elastic_tuple, tuple)
        self.assertIsInstance(elastic_params, dict)
        self.assertIsInstance(elastic_mod, self.Pipeline)
        self.assertIsInstance(elastic_mod.steps[-1][1], self.ElasticNet)

    @patch('ml_program.Regressors.create_ann')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_lasso')
    @patch('ml_program.Regressors.create_linear')
    def test_create_svr(self, mock_linear, mock_lasso,
                        mock_ridge, mock_elastic, mock_ann):
        """make sure create_svr return pipeline obj w SVR model"""
        mock_linear.return_value = 'Linear Regression'
        mock_lasso.return_value = 'Lasso Regression'
        mock_ridge.return_value = 'Ridge Regression'
        mock_elastic.return_value = 'Elastic Net Regression'
        mock_ann.return_value = 'Ann Model'
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_NUMERICAL, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Regressors.create_svr',
                   return_value='False'):
            regressor_object = Regressors(data_obj)
        svr_tuple = regressor_object.create_svr(data_obj)
        svr_mod = svr_tuple[0]
        svr_params = svr_tuple[1]
        self.assertIsInstance(svr_tuple, tuple)
        self.assertIsInstance(svr_params, dict)
        self.assertIsInstance(svr_mod, self.Pipeline)
        self.assertIsInstance(svr_mod.steps[-1][1], self.SVR)

    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_lasso')
    @patch('ml_program.Regressors.create_linear')
    def test_create_ann(self, mock_linear, mock_lasso,
                        mock_ridge, mock_elastic, mock_svr):
        """test that create ann return a CreateAnn object for regressor"""
        mock_linear.return_value = 'Linear model'
        mock_lasso.return_value = 'Lasso model'
        mock_ridge.return_value = 'Ridge model'
        mock_elastic.return_value = 'Elastic model'
        mock_svr.return_value = 'SVR model'
        import tensorflow as tf
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input',
                   side_effect=[csv_file,
                                COLUMN_NUMERICAL_CAT_DATA, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Regressors.create_ann',
                   return_value='False'):
            regressor_object = Regressors(data_obj)
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_mod = regressor_object.create_ann(data_obj)
        self.assertIsInstance(ann_mod[0].model, tf.keras.models.Sequential)
        self.assertTrue(ann_mod[0].loss == 'mse')

    @patch('ml_program.Regressors.create_svr')
    @patch('ml_program.Regressors.create_elastic')
    @patch('ml_program.Regressors.create_ridge')
    @patch('ml_program.Regressors.create_lasso')
    @patch('ml_program.Regressors.create_linear')
    def test_create_ann_false(self, mock_linear, mock_lasso,
                              mock_ridge, mock_elastic, mock_svr):
        """test that create_ann return False if model creation fail"""
        mock_linear.return_value = 'Linear model'
        mock_lasso.return_value = 'Lasso model'
        mock_ridge.return_value = 'Ridge model'
        mock_elastic.return_value = 'Elastic model'
        mock_svr.return_value = 'SVR model'
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NUMERICAL_CAT_DATA,
                                                  '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Regressors.create_ann',
                   return_value='False'):
            regressor_object = Regressors(data_obj)
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'not activationfunk'
        optimizer = 'adm'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'Fal'
        monitor = 'accuray'
        mode = 'mn'
        patience = '5'
        verbose = 'nope'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                regressor_object.create_ann(data_obj)
                self.assertFalse(regressor_object.ann_model)

    def test_score_models(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input',
                   side_effect=[csv_file,
                                COLUMN_NUMERICAL_CAT_DATA, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
            layer_tuple = (2, -0.22, 30, 4)
            activation = 'relu'
            optimizer = 'adam'
            batch_size = '10'
            epochs = '2'
            multiprocess = 'False'
            monitor = 'accuracy'
            mode = 'min'
            patience = '2'
            verbose = '1'
            with patch('ml_program.CreateANN.choose_value',
                       side_effect=[activation, optimizer,
                                    batch_size, epochs,
                                    multiprocess, verbose,
                                    monitor, mode, patience]):
                with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                           return_value=layer_tuple):
                    model_class = Regressors(data_obj)
        linear = model_class.linear_tuple
        lasso = model_class.lasso_tuple
        ridge = model_class.ridge_tuple
        elastic = model_class.elastic_tuple
        svr = model_class.svr_tuple
        ann = model_class.ann_tuple
        self.assertIsInstance(linear, tuple)
        self.assertIsInstance(lasso, tuple)
        self.assertIsInstance(ridge, tuple)
        self.assertIsInstance(elastic, tuple)
        self.assertIsInstance(svr, tuple)
        self.assertIsInstance(ann, tuple)
        self.assertTrue(len(ann) == 2)
        self.assertIsInstance(ann[0], CreateANN)
        regressor_scores = model_class.score_models()
        self.assertIsInstance(regressor_scores, list)
        self.assertEqual(len(regressor_scores), 6)
        for model in regressor_scores:
            self.assertIsInstance(model, dict)

    def test_display_model_evaluation_default_model(self):
        """confirm that method return a model if no specific is selected"""
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input',
                   side_effect=[csv_file,
                                COLUMN_NAME_NUMERICAL, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '2'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '2'
        verbose = '1'
        with patch('ml_program.CreateANN.choose_value',
                   side_effect=[activation, optimizer,
                                batch_size, epochs,
                                multiprocess, verbose,
                                monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                regressor_class = Regressors(data_obj)
        regressor_class.score_models()
        with patch('builtins.input', return_value='x'):
            self.assertIsInstance(
                regressor_class.display_model_evaluation(),
                (self.Pipeline, CreateANN))

    def test_display_model_evaluation_select_model(self):
        """confirm specicfied model i returned"""
        csv_file = NUMERICAL_NO_MISSING
        with patch('builtins.input',
                   side_effect=[csv_file, COLUMN_NAME_NUMERICAL, '1']):
            with patch('ml_program.Regressors', return_value='Regressor'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '2'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '2'
        verbose = '1'
        with patch('ml_program.CreateANN.choose_value',
                   side_effect=[activation, optimizer,
                                batch_size, epochs,
                                multiprocess, verbose,
                                monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                regressor_class = Regressors(data_obj)
        regressor_class.score_models()
        with patch('builtins.input', return_value='LinearRegression'):
            self.assertIsInstance(
                regressor_class.display_model_evaluation().steps[-1][1],
                self.LinearRegression)


class Test_Classifiers(TestCase):
    from sklearn.linear_model import LogisticRegression as LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
    from sklearn.svm import SVC as SVC
    from sklearn.model_selection import GridSearchCV as GridSearchCV
    from sklearn.pipeline import Pipeline as Pipeline

    @patch('ml_program.Classifiers.create_ann')
    @patch('ml_program.Classifiers.create_svc')
    @patch('ml_program.Classifiers.create_knn')
    @patch('ml_program.Classifiers.create_logistic')
    def test_constructor(self, mock_logistic, mock_knn, mock_svc,
                         mock_ann):
        mock_logistic.return_value = 'Logistic'
        mock_knn.return_value = 'Knn'
        mock_svc.return_value = 'Svc'
        mock_ann.return_value = 'ann'
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        classifier_object = Classifiers(data_obj)
        self.assertEqual(data_obj, classifier_object.dataobject)
        self.assertIsInstance(classifier_object, Classifiers)
        self.assertIsInstance(classifier_object.models, list)

    @patch('ml_program.Classifiers.create_ann')
    @patch('ml_program.Classifiers.create_svc')
    @patch('ml_program.Classifiers.create_knn')
    @patch('ml_program.Classifiers.create_logistic')
    def test_get_pipeline_grid(self, mock_logitsic,
                               mock_knn, mock_svc, mock_ann):
        """assure get_pipeline_Grid in Classifier retun pipeline obj"""
        csv_file = CATEGORICAL_NO_MISSING
        mock_logitsic.return_value = 'Logistic Classifier'
        mock_knn.return_value = 'Knn model'
        mock_svc.return_value = 'svc model'
        mock_ann.return_value = 'Ann Model'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
                classifier_obj = Classifiers(data_obj)
        param_grid = {'model__C': [0.001, 0.1, 10, 100, 10e5],
                      'model__penalty': ['l2']}
        self.assertIsInstance(classifier_obj.get_pipeline_grid(
            self.LogisticRegression, param_grid),
            self.GridSearchCV)

    @patch('ml_program.Classifiers.create_ann')
    @patch('ml_program.Classifiers.create_svc')
    @patch('ml_program.Classifiers.create_knn')
    def test_create_logistic(self, mock_knn, mock_svc, mock_ann):
        """test logistic classifier metohod"""
        csv_file = CATEGORICAL_NO_MISSING
        mock_knn.return_value = 'Knn Model'
        mock_svc.return_value = 'Svc Model'
        mock_ann.return_value = 'Ann Model'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Classifiers.create_logistic',
                   return_value='False'):
            classifier_object = Classifiers(data_obj)
        logistic_tuple = classifier_object.create_logistic(data_obj)
        self.assertIsInstance(logistic_tuple, tuple)
        logistic_mod = logistic_tuple[0]
        logistic_params = logistic_tuple[1]
        self.assertIsInstance(logistic_mod, self.Pipeline)
        self.assertIsInstance(logistic_mod.steps[-1][1],
                              self.LogisticRegression)
        self.assertIsInstance(logistic_params, dict)

    @patch('ml_program.Classifiers.create_ann')
    @patch('ml_program.Classifiers.create_svc')
    @patch('ml_program.Classifiers.create_logistic')
    def test_create_knn_(self, mock_logistic, mock_svc, mock_ann):
        """test create_knn returns knn model in pipeline"""
        csv_file = CATEGORICAL_NO_MISSING
        mock_logistic.return_value = 'Knn Model'
        mock_svc.return_value = 'Svc Model'
        mock_ann.return_value = 'Ann Model'
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Classifiers.create_knn',
                   return_value='False'):
            classifier_object = Classifiers(data_obj)
        knn_tuple = classifier_object.create_knn(data_obj)
        self.assertIsInstance(knn_tuple, tuple)
        knn_mod = knn_tuple[0]
        knn_params = knn_tuple[1]
        self.assertIsInstance(knn_params, dict)
        self.assertIsInstance(knn_mod, self.Pipeline)
        self.assertIsInstance(knn_mod.steps[-1][1],
                              self.KNeighborsClassifier)

    @patch('ml_program.Classifiers.create_ann')
    @patch('ml_program.Classifiers.create_knn')
    @patch('ml_program.Classifiers.create_logistic')
    def test_create_svc(self, mock_logistic, mock_knn, mock_ann):
        """test create_svc- asssure it returns pipeline with svc model"""
        mock_logistic.return_value = 'Logistic Classifier'
        mock_knn.return_value = 'Knn model'
        mock_ann.return_value = 'Ann Model'
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Classifiers.create_svc',
                   return_value='False'):
            classifier_object = Classifiers(data_obj)
        svc_tuple = classifier_object.create_svc(data_obj)
        self.assertIsInstance(svc_tuple, tuple)
        svc_mod = svc_tuple[0]
        svc_params = svc_tuple[1]
        self.assertIsInstance(svc_params, dict)
        self.assertIsInstance(svc_mod, self.Pipeline)
        self.assertIsInstance(svc_mod.steps[-1][1],
                              self.SVC)

    @patch('ml_program.Classifiers.create_svc')
    @patch('ml_program.Classifiers.create_knn')
    @patch('ml_program.Classifiers.create_logistic')
    def test_create_ann_multiclass(self, mock_logistic, mock_knn, mock_svc):
        """test that create ann return a CreateAnn model for multiclass"""
        mock_logistic.return_value = 'Logistic Classifier'
        mock_knn.return_value = 'Knn model'
        mock_svc.return_value = 'svc Model'
        import tensorflow as tf
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Classifiers.create_ann',
                   return_value='False'):
            classifier_object = Classifiers(data_obj)
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_mod = classifier_object.create_ann(data_obj)
        self.assertIsInstance(ann_mod[0].model, tf.keras.models.Sequential)
        self.assertTrue(ann_mod[0].model.loss == 'categorical_crossentropy')

    @patch('ml_program.Classifiers.create_svc')
    @patch('ml_program.Classifiers.create_knn')
    @patch('ml_program.Classifiers.create_logistic')
    def test_create_ann_binary(self, mock_logistic, mock_knn, mock_svc):
        """test that create ann return a Ann model for binary classification"""
        mock_logistic.return_value = 'Logistic Classifier'
        mock_knn.return_value = 'Knn model'
        mock_svc.return_value = 'svc Model'
        import tensorflow as tf
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_BINARYCLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Classifiers.create_ann',
                   return_value='False'):
            classifier_object = Classifiers(data_obj)
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                ann_mod = classifier_object.create_ann(data_obj)
        self.assertIsInstance(ann_mod[0].model, tf.keras.models.Sequential)
        self.assertTrue(ann_mod[0].model.loss == 'binary_crossentropy')

    @patch('ml_program.Classifiers.create_svc')
    @patch('ml_program.Classifiers.create_knn')
    @patch('ml_program.Classifiers.create_logistic')
    def test_create_ann_classifier_fail(self, mock_logistic,
                                        mock_knn, mock_svc):
        """test that create ann return a Ann model for binary classification"""
        mock_logistic.return_value = 'Logistic Classifier'
        mock_knn.return_value = 'Knn model'
        mock_svc.return_value = 'svc Model'
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_BINARYCLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        with patch('ml_program.Classifiers.create_ann',
                   return_value=False):
            classifier_object = Classifiers(data_obj)
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adm'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuray'
        mode = 'mn'
        patience = '5'
        verbose = '22'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                classifier_object.create_ann(data_obj)
        self.assertFalse(classifier_object.ann_model)

    def test_score_models_binary(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_BINARYCLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                classifier_object = Classifiers(data_obj)
        scored_models = classifier_object.score_models()
        self.assertIsInstance(scored_models, list)
        self.assertEqual(len(scored_models), 4)
        for model in scored_models:
            self.assertIsInstance(model, dict)

    def test_score_models_multiclass(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                classifier_object = Classifiers(data_obj)
        scored_models = classifier_object.score_models()
        self.assertIsInstance(scored_models, list)
        self.assertEqual(len(scored_models), 4)
        for model in scored_models:
            self.assertIsInstance(model, dict)

    def test_display_model_evaluation_best_model(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                classifier_object = Classifiers(data_obj)
        classifier_object.score_models()
        with patch('builtins.input', return_value='x'):
            self.assertIsInstance(
                classifier_object.display_model_evaluation(),
                (self.Pipeline, CreateANN))

    def test_display_model_evaluation_choosen_model(self):
        csv_file = CATEGORICAL_NO_MISSING
        with patch('builtins.input', side_effect=[csv_file,
                                                  COLUMN_NAME_MULTICLASS,
                                                  '2']):
            with patch('ml_program.Classifiers', return_value='Classifier'):
                model_creator = CreateModel()
                data_obj = model_creator.dataobj
        layer_tuple = (2, -0.22, 30, 4)
        activation = 'relu'
        optimizer = 'adam'
        batch_size = '10'
        epochs = '22'
        multiprocess = 'False'
        monitor = 'accuracy'
        mode = 'min'
        patience = '5'
        verbose = '1'
        with patch('builtins.input', side_effect=[activation, optimizer,
                                                  batch_size, epochs,
                                                  multiprocess, verbose,
                                                  monitor, mode, patience]):
            with patch('ml_program.CreateANN.define_hidden_layer_tuple',
                       return_value=layer_tuple):
                classifier_object = Classifiers(data_obj)
        classifier_object.score_models()
        with patch('builtins.input', return_value='LogisticRegression'):
            self.assertIsInstance(
                classifier_object.display_model_evaluation().steps[-1][1],
                self.LogisticRegression)
