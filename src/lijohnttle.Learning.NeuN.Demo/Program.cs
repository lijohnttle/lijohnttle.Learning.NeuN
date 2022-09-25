using lijohnttle.Learning.NeuN.Core;

var neuralNetwork = new NeuralNetwork(3, 3, 3, 0.3);

var inputs = new double[]
{
    1,
    2,
    3
};


var result = neuralNetwork.Query(inputs);

Console.WriteLine("Hello, World!");
