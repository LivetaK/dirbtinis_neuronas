import math
import time

class Neuron:


    def calculate_a(self, x, weights, bias):
        return sum(xi * wi for xi, wi in zip(x, weights)) + bias

    def calculate_sigmoid_output(self, x, weights, bias):
        a = self.calculate_a(x, weights, bias)
        return 1 / (1 + math.exp(-a))

    def accuracy(self, answ_df):
        correct_predictions = 0
        for label, pred in zip(answ_df['label'], answ_df['predicted_label']):
            if label == pred:
                correct_predictions += 1
        return correct_predictions / len(answ_df)

    def calculate_err_acc(self, data, w, bias):
        data = data.copy()
        columns = [c for c in data.columns if c != 'label']
        data['predicted_label'] = None
        total_error = 0

        for i, row in data.iterrows():
            x = []
            for col in columns:
                x.append(row[col])
            t = row['label']
            y = self.calculate_sigmoid_output(x, w, bias)
            data.at[i, 'predicted_label'] = round(y)
            error = (t - y) ** 2
            total_error += error
        accuracy = self.accuracy(data[['label', 'predicted_label']])
        return total_error / len(data), accuracy, data
        
    def stochastic_gradient_descent(self,train_data, validation_data, test_data, w, bias, learning_rate, epochs, emin):
        columns = [c for c in train_data.columns if c != 'label']
        train_error = float('inf')

        train_errors = []
        train_accuracies = []
        validation_errors = []
        validation_accuracies = []

        best_epoch = {}
        best_epoch["score"] = float('-inf')

        epoch = 0
        start_time = time.time()

        while train_error > emin and epoch < epochs:
            train_data['predicted_label']  = None
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            train_error = 0

            for i, row in train_data.iterrows():
                x = []

                for col in columns:
                    x.append(row[col])
                t = row['label']
                y = self.calculate_sigmoid_output(x, w, bias)
                train_data.at[i, 'predicted_label'] = round(y)

                for k in range(len(columns)):
                    w[k] -= learning_rate * (y - t) * y * (1 - y) * x[k]
                bias -= learning_rate * (y - t) * y * (1 - y)

                error = (t - y) ** 2
                train_error += error
            average_train_error = train_error / len(train_data)
            train_errors.append(average_train_error)
            train_accuracie = self.accuracy(train_data[['label', 'predicted_label']])
            train_accuracies.append(train_accuracie)

            validation_error, validation_accuracy, _ = self.calculate_err_acc(validation_data, w, bias)
            validation_errors.append(validation_error)
            validation_accuracies.append(validation_accuracy)

            score = validation_accuracy - validation_error
            if score > best_epoch["score"]:
                best_epoch["score"] = score
                best_epoch["learning_rate"] = learning_rate
                best_epoch["epoch_indx"] = epoch
                best_epoch["train_error"] = average_train_error
                best_epoch["validation_error"] = validation_error
                best_epoch["train_accuracy"] = train_accuracie
                best_epoch["validation_accuracy"] = validation_accuracy
                best_epoch["weights"] = w.copy()
                best_epoch["bias"] = bias
            epoch += 1
        
        training_time = time.time() - start_time
        test_error, test_accuracy, test_calc_df = self.calculate_err_acc(test_data, best_epoch["weights"], best_epoch["bias"])


        return w, bias, train_errors, train_accuracies, validation_errors, validation_accuracies, test_error, test_accuracy, test_calc_df, training_time, best_epoch

    def batch_gradient_descent(self,train_data, validation_data, test_data, w, bias, learning_rate, epochs, emin):     
        columns = [c for c in train_data.columns if c != 'label']
        train_error = float('inf')

        train_errors = []
        train_accuracies = []
        validation_errors = []
        validation_accuracies = []

        best_epoch = {}
        best_epoch["score"] = float('-inf')

        epoch = 0
        start_time = time.time()

        while train_error > emin and epoch < epochs:
            train_data['predicted_label']  = None
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            train_error = 0
            gradientSum = [0.0] * len(columns)
            biasSum = 0.0


            for i, row in train_data.iterrows():
                x = []

                for col in columns:
                    x.append(row[col])
                t = row['label']
                y = self.calculate_sigmoid_output(x, w, bias)
                train_data.at[i, 'predicted_label'] = round(y)

                for k in range(len(columns)):
                    gradientSum[k] += (y - t) * y * (1 - y) * x[k]

                biasSum += (y - t) * y * (1 - y)

                error = (t - y) ** 2
                train_error += error
            for k in range(len(columns)):
                w[k] -= learning_rate * (gradientSum[k] / len(train_data))
            bias -= learning_rate * (biasSum / len(train_data))
            train_errors.append(train_error / len(train_data))
            train_accuracie = self.accuracy(train_data[['label', 'predicted_label']])
            train_accuracies.append(train_accuracie)

            validation_error, validation_accuracy, _ = self.calculate_err_acc(validation_data, w, bias)
            validation_errors.append(validation_error)
            validation_accuracies.append(validation_accuracy)
            
            score = validation_accuracy - validation_error
            if score > best_epoch["score"]:
                best_epoch["score"] = score
                best_epoch["learning_rate"] = learning_rate
                best_epoch["epoch_indx"] = epoch
                best_epoch["train_error"] = train_error / len(train_data)
                best_epoch["validation_error"] = validation_error
                best_epoch["train_accuracy"] = train_accuracie
                best_epoch["validation_accuracy"] = validation_accuracy
                best_epoch["weights"] = w.copy()
                best_epoch["bias"] = bias

            epoch += 1

        training_time = time.time() - start_time
        test_error, test_accuracy, test_calc_df = self.calculate_err_acc(test_data, best_epoch["weights"], best_epoch["bias"])

        return w, bias, train_errors, train_accuracies, validation_errors, validation_accuracies, test_error, test_accuracy, test_calc_df, training_time, best_epoch