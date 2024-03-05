using System;

namespace NeuralNetwork;

public class Program
{
    public static void Main(string[] args)
    {
        Network n = new Network()
            .AddLayer(2, 4, ActivationFunction.Sigmoid)
            .AddLayer(4, 1, ActivationFunction.Sigmoid)
            .RandomizeWeights();
            
        var set = new TrainingSet(new () {
            new TrainingItem(new double[] {0, 0}, new double[] {0}),
            new TrainingItem(new double[] {1, 0}, new double[] {1}),
            new TrainingItem(new double[] {0, 1}, new double[] {1}),
            new TrainingItem(new double[] {1, 1}, new double[] {0}),
        });

        var trainer = new BackpropagationTrainer(n);
        trainer.Learn(set, 1000000, 0.9);

        foreach(var item in set.Items)
        {
            Console.WriteLine(n.Feed(item.Input)[0]);
        }

        Console.WriteLine(n.GetNeuronState());
        Console.WriteLine(n.GetWeightState());
    }
}