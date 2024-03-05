using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class TrainingSet
    {
        public List<TrainingItem> Items { get; private set; }

        public TrainingSet(List<TrainingItem> items)
        {
            Items = items;
        }
    }

    public class TrainingItem
    {
        public double[] Input { get; init; }
        public double[] ExpectedOutput { get; init; }

        public TrainingItem(double[] input, double[] expectedOutput)
        {
            Input = input;
            ExpectedOutput = expectedOutput;
        }
    }

    public class BackpropagationTrainer
    {
        private Network _network;
        public BackpropagationTrainer(Network network)
        {
            this._network = network;
        }

        public void Learn(TrainingSet set, int epochs, double learningCoefficient)
        {
            for(int i = 0; i < epochs; i++)
            {
                foreach(var item in set.Items)
                {
                    Backpropagate(_network, item.Input, item.ExpectedOutput, learningCoefficient);
                }
            }
        }

        private static double[] Backpropagate(Network network, double[] input, double[] expectedOutput, double learningCoefficient)
        {
            var output = network.Feed(input);
            var delta = GetDelta(output, expectedOutput);

            var deltaHolder = network.Neurons.CopyNeuronSignature();

            for(int layerIndex = network.Neurons.Length - 1; layerIndex >= 0; layerIndex--)
            {
                var lastLayer = layerIndex == network.Neurons.Length - 1;
                var function = layerIndex > 0 ? network.Functions[layerIndex - 1] : ActivationFunction.Identity;

                if(lastLayer)
                {
                    for(int outputIndex = 0; outputIndex < network.Neurons[layerIndex].Length; outputIndex++)
                    {
                        var d = delta[outputIndex];
                        var raw = network.RawNeurons[layerIndex][outputIndex];
                        var der = function.Derivative(raw);

                        var newDelta = d * der;
                        deltaHolder[layerIndex][outputIndex] = newDelta;
                    }

            
                    continue;
                }

                for(int neuronIndex = 0; neuronIndex < network.Neurons[layerIndex].Length; neuronIndex++)
                {
                    var raw = network.RawNeurons[layerIndex][neuronIndex];
                    var der = function.Derivative(raw);

                    var outcomingDeltaSum = 0.0;
                    for(int outcomingNeuron = 0; outcomingNeuron < network.GetLayerOutputs(layerIndex); outcomingNeuron++)
                    {
                        var weight = network.Weights[layerIndex][neuronIndex, outcomingNeuron];
                        var d = deltaHolder[layerIndex + 1][outcomingNeuron];

                        outcomingDeltaSum += weight * d;
                    }

                    deltaHolder[layerIndex][neuronIndex] = der * outcomingDeltaSum;

                    
                    for(int outcomingNeuron = 0; outcomingNeuron < network.GetLayerOutputs(layerIndex); outcomingNeuron++)
                    {
                        var thisOutput = network.Neurons[layerIndex][neuronIndex];
                        var receiverDelta = deltaHolder[layerIndex + 1][outcomingNeuron];
                        var weight = network.Weights[layerIndex][neuronIndex, outcomingNeuron];

                        var gradient = thisOutput * receiverDelta;

                        network.Weights[layerIndex][neuronIndex, outcomingNeuron] += gradient * learningCoefficient;
                    }
                }
            }

            return output;
        }

        public static double[] GetDelta(double[] realOutput, double[] expectedOutput)
        {
            double[] result = realOutput.Select((value, index) => - value + expectedOutput[index]).ToArray();
            return result;
        }

        public static double[] GetSquareDelta(double[] realOutput, double[] expectedOutput)
        {
            double[] result = GetDelta(realOutput, expectedOutput).Select(v => v*v).ToArray();
            return result;
        }

        public static double[] GetSquareHalfDelta(double[] realOutput, double[] expectedOutput)
        {
            double[] result = GetDelta(realOutput, expectedOutput).Select(v => v*v*0.5).ToArray();
            return result;
        }

        public static double[] GetError(double[] realOutput, double[] expectedOutput)
        {
            double[] result = GetDelta(realOutput, expectedOutput).Select(v => Math.Abs(v)).ToArray();
            return result;
        }

        public static double GetTotalError(double[] realOutput, double[] expectedOutput)
        {
            return GetError(realOutput, expectedOutput).Sum();
        }
    }
}