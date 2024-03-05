using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public static class ArrayUtilities
    {
        public static double[][] CopyNeuronSignature(this double[][] neurons)
        {
            var array = new double[neurons.Length][];
            
            for(int i = 0; i < neurons.Length; i++)
            {
                array[i] = new double[neurons[i].Length];
            }
            return array;
        }
    }
}