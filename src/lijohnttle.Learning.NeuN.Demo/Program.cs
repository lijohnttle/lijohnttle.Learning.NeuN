using lijohnttle.Learning.NeuN.Core;

var inputLayerSize = 784;
var hiddenLayerSize = 100;
var outputLayerSize = 10;
var learningRate = 0.3;

var neuralNetwork = new NeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize, learningRate);

/*
 * The list of commands:
 * - train - Looking for *.csv files in the ./train folder
 * - test - Looking for *.csv files in the ./test folder
 */

while (true)
{
    Console.WriteLine("Enter command to continue (enter exit to quit):");

    var command = Console.ReadLine();

    switch (command)
    {
        case "train":
            foreach (var file in Directory.GetFiles("./train", "*.csv"))
            {
                using (var fileReader = File.OpenRead(file))
                using (var textReader = new StreamReader(fileReader))
                {
                    string? sample;

                    while ((sample = textReader.ReadLine()) != null)
                    {
                        var tokens = sample.Split(',');
                        var label = tokens[0].Length == 0 ? 0 : int.Parse(tokens[0]);

                        // get pixels values in the range 0.01-1.00
                        var inputs = tokens.Skip(1).Select(t => (int.Parse(t) / 255.0) * 0.99 + 0.01).ToArray();
                        var targets = BuildTargets(label);

                        neuralNetwork.Train(inputs, targets);
                    }
                }
            }

            break;

        case "test":
            int successCounter = 0;
            int failCounter = 0;

            foreach (var file in Directory.GetFiles("./test", "*.csv"))
            {
                using (var fileReader = File.OpenRead(file))
                using (var textReader = new StreamReader(fileReader))
                {
                    string? sample;

                    while ((sample = textReader.ReadLine()) != null)
                    {
                        var tokens = sample.Split(',');
                        var label = tokens[0].Length == 0 ? 0 : int.Parse(tokens[0]);

                        // get pixels values in the range 0.01-1.00
                        var inputs = tokens.Skip(1).Select(t => (int.Parse(t) / 255.0) * 0.99 + 0.01).ToArray();

                        var outputs = neuralNetwork.Query(inputs);

                        var result = outputs
                            .Select((t, i) => new { Label = i, Confidence = t })
                            .OrderByDescending(t => t.Confidence)
                            .First();

                        var success = label == result.Label;
                        var status = success ? "Success" : "Failure";

                        Console.WriteLine($"Status: {status}, Expected: {label}, Actual: {result.Label}, Confidence: {result.Confidence}");

                        if (success)
                        {
                            successCounter++;
                        }
                        else
                        {
                            failCounter++;
                        }
                    }
                }
            }

            Console.WriteLine($"Success Rate: {successCounter / (double)(successCounter + failCounter) * 100}%");
            break;

        case "exit":
            return;
    }
}

double[] BuildTargets(int label)
{
    return Enumerable.Range(0, 10).Select((t, i) => i == label ? 0.99 : 0.01).ToArray();
}
