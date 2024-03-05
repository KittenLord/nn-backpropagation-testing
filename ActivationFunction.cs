using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class ActivationFunction
    {
        public static readonly ActivationFunction Identity = new ActivationFunction("-", x => x, x => 1);
        public static readonly ActivationFunction ReLU = new ActivationFunction("ReLU", x => x > 0 ? x : 0, x => x > 0 ? 1 : 0);
        public static readonly ActivationFunction Sigmoid = new ActivationFunction("Sigma", x => _Sigmoid(x), x => _Sigmoid(x)*(1 - _Sigmoid(x)));

        private static double _Sigmoid(double x) => 1 / (1 + Math.Exp(-x));



        public string Name { get; init; }
        public Func<double, double> Function { get; init; }
        public Func<double, double> Derivative { get; init; }

        public ActivationFunction(string name, Func<double, double> function, Func<double, double> derivative)
        {
            Name = name;
            Function = function;
            Derivative = derivative;
        }
    }
}