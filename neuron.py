import math


class Neuron:

    def calculate_a(self, x, weights, bias):
        return sum(xi * wi for xi, wi in zip(x, weights)) + bias

    def calculate_step_output(self, x, weights, bias):
        a = self.calculate_a(x, weights, bias)
        return 1 if a >= 0 else 0
    
    def calculate_sigmoid_output(self, x, weights, bias):
        a = self.calculate_a(x, weights, bias)
        ans = round(1 / (1 + math.exp(-a)))
        return ans
        
    def calculate_output_for_df(self, df, weights, bias, method):
        outputs = []
        for i, row in df.iterrows():
            x = [row['x'], row['y']]
            if method == 'sigmoid':
                output = self.calculate_sigmoid_output(x, weights, bias)
            elif method == 'step':
                output = self.calculate_step_output(x, weights, bias)
            else:
                print('wrong input')
            outputs.append(output)
        return outputs