using MathNet.Numerics.LinearAlgebra;

namespace lijohnttle.Learning.NeuN.Core.Internal
{
    internal class QueryProcessor : IQueryProcessor
    {
        public static QueryProcessor Default { get; } = new QueryProcessor();


        public Vector<double>[] Query(
            Vector<double> inputs,
            Matrix<double>[] weightsLayers,
            Func<double, double> activationFunction)
        {
            var outputs = new List<Vector<double>>();
            var previousOutputs = inputs;

            foreach (var weightsLayer in weightsLayers)
            {
                var actualInputs = weightsLayer * previousOutputs;
                var actualOutputs = actualInputs.Map(activationFunction);

                previousOutputs = actualOutputs;
                outputs.Add(actualOutputs);
            }

            return outputs.ToArray();
        }
    }
}
