using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Network
    {
        public int LayerCount = 0;

        public ActivationFunction[] Functions = new ActivationFunction[0];
        public double[][] RawNeurons { get; private set; }
        public double[][] Neurons { get; private set; }
        public double[][,] Weights { get; private set; } = new double[0][,];

        public int GetLayerInputs(int layer) => Weights[layer].GetLength(0);
        public int GetLayerOutputs(int layer) => Weights[layer].GetLength(1);

        public Network()
        {
            RawNeurons = new double[1][];
            Neurons = new double[1][];
        }

        public Network AddLayer(int inputs, int outputs, ActivationFunction activationFunction)
        {
            RawNeurons = RawNeurons.SkipLast(1).ToArray();
            Neurons = Neurons.SkipLast(1).ToArray();

            LayerCount++;
            Functions = Functions.Append(activationFunction).ToArray();
            RawNeurons = RawNeurons.Append(new double[inputs]).ToArray();
            Neurons = Neurons.Append(new double[inputs]).ToArray();
            Weights = Weights.Append(new double[inputs, outputs]).ToArray();

            RawNeurons = Neurons.Append(new double[outputs]).ToArray();
            Neurons = Neurons.Append(new double[outputs]).ToArray();

            return this;
        }

        public Network RandomizeWeights()
        {
            for(int i = 0; i < Weights.Length; i++)
            {
                var layer = Weights[i];
                for(int j = 0; j < layer.GetLength(0); j++)
                {
                    for(int k = 0; k < layer.GetLength(1); k++)
                    {
                        Weights[i][j,k] = RandomGen.Double2();
                    }
                }
            }

            return this;
        }

        private void Clear()
        {
            for(int layer = 0; layer < Neurons.Length; layer++)
            {
                Neurons[layer] = Neurons[layer].Select(v => 0.0).ToArray();
                RawNeurons[layer] = RawNeurons[layer].Select(v => 0.0).ToArray();
            }
        }

        public double[] Feed(double[] inputData)
        {
            Clear();

            inputData.CopyTo(RawNeurons[0], 0);
            inputData.CopyTo(Neurons[0], 0);

            for(int layer = 0; layer < LayerCount; layer++)
            {
                var function = Functions[layer];
                for(int output = 0; output < GetLayerOutputs(layer); output++)
                {
                    for(int input = 0; input < GetLayerInputs(layer); input++)
                    {
                        var weight = Weights[layer][input, output];
                        var neuronValue = Neurons[layer][input];

                        var value = weight * neuronValue;

                        RawNeurons[layer + 1][output] += value;
                    }

                    Neurons[layer + 1][output] = function.Function(RawNeurons[layer + 1][output]);
                }
            }

            return Neurons.Last();
        }

        public string GetNeuronState()
        {
            string str = "";
            for(int layer = 0; layer < Neurons.Length; layer++)
            {
                str += string.Join(" ", Neurons[layer]);
                if (layer != Neurons.Length - 1) 
                {
                    str += "\n";
                    str += Functions[layer].Name;
                    str += "\n";
                }
            }   
            return str;
        }

        public string GetWeightState()
        {
            string str = "";
            for(int layer = 0; layer < LayerCount; layer++)
            {
                str += $"layer {layer}: ";
                for(int i = 0; i < GetLayerInputs(layer); i++)
                {
                    for(int j = 0; j < GetLayerOutputs(layer); j++)
                    {
                        str += $"w{i}{j} = {Weights[layer][i,j]};  ";
                    }
                }
                str += "\n";
            }
            return str;
        }
    }
}