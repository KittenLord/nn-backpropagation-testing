using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public static class RandomGen
    {
        private static Random r;

        static RandomGen()
        {
            r = new Random();
        }

        public static int Int() => r.Next();
        public static double Double() => r.NextDouble();
        public static double Double2() => r.NextDouble() * 2 - 1;
        public static double Double05() => r.NextDouble() - 0.5;
    }
}