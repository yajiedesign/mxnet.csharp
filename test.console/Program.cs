using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using log4net.Config;
using mxnet.csharp;
using mxnet.csharp.callback;
using mxnet.csharp.initializer;
using mxnet.csharp.metric;
using mxnet.csharp.optimizer;
using mxnet.csharp.util;
using mxnet.numerics.nbase;
using mxnet.numerics.single;
using Shape = mxnet.csharp.Shape;

namespace test.console
{
    class Program
    {

        static Symbol get_ocrnet(int batchSize)
        {
            using (NameScop scop = new NameScop())
            {
                var data = Symbol.Variable("data");
                var label = Symbol.Variable("softmax_label");
                var conv1 = Symbol.Convolution(data, kernel: new Shape(5, 5), num_filter: 32);
                var pool1 = Symbol.Pooling(conv1, pool_type: PoolingPoolType.Max, kernel: new Shape(2, 2),
                    stride: new Shape(1, 1));
                var relu1 = Symbol.Activation(data: pool1, act_type: ActivationActType.Relu);

                var conv2 = Symbol.Convolution(relu1, kernel: new Shape(5, 5), num_filter: 32);
                var pool2 = Symbol.Pooling(data: conv2, pool_type: PoolingPoolType.Avg,
                    kernel: new Shape(2, 2), stride: new Shape(1, 1));
                var relu2 = Symbol.Activation(data: pool2, act_type: ActivationActType.Relu);
                var conv3 = Symbol.Convolution(data: relu2, kernel: new Shape(3, 3), num_filter: 32);
                var pool3 = Symbol.Pooling(data: conv3, pool_type: PoolingPoolType.Avg,
                    kernel: new Shape(2, 2), stride: new Shape(1, 1));
                var relu3 = Symbol.Activation(data: pool3, act_type: ActivationActType.Relu);
                var flatten = Symbol.Flatten(data: relu3);
                var fc1 = Symbol.FullyConnected(data: flatten, num_hidden: 512);
                var fc21 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc22 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc23 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc24 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc2 = Symbol.Concat(new Symbol[] { fc21, fc22, fc23, fc24 }, 4, dim: 0);
                label = Symbol.Transpose(data = label);
                label = Symbol.Reshape(data = label, shape: new Shape((uint)(batchSize * 4)));
                return Symbol.SoftmaxOutput("softmax", fc2, label);
            }
        }

        static void Main(string[] args)
        {

            NumericsTest test = new NumericsTest();
            test.Test();

            var log4NetConfig = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), "log4net.config");
            XmlConfigurator.Configure(new FileInfo(log4NetConfig));


            //var model = FeedForward.Load("checkpoint\\cnn");

            //ReadData rdpredict = new ReadData("data\\train\\", 32, true);
            //var testOut = model.Predict(rdpredict, 1);


            TrainTest();
        }

        private static void TrainTest()
        {
            int batch_size = 32;
            var pnet = get_ocrnet(batch_size);
            //var modelloadtest = FeedForward.Load("checkpoint\\cnn");



            ReadData rdtrain = new ReadData("data\\train\\", batch_size);
            ReadData rdval = new ReadData("data\\val\\", batch_size);

            Context ctx = new Context(DeviceType.KGpu, 0);


            Speedometer speed = new Speedometer(batch_size, 50);
            DoCheckpoint doCheckpoint = new DoCheckpoint("checkpoint\\cnn");

            CustomMetric customMetric = new CustomMetric((l, p) => Accuracy(l, p, batch_size), "Accuracy");

            Optimizer optimizer = new CcSgd(momentum: 0.9f, learningRate: 0.001f, wd: 0.00001f, rescaleGrad: 1.0f / batch_size);

            FeedForward model = null;
            try
            {
                var modelload = FeedForward.Load("checkpoint\\cnn", ctx: ctx,
                    numEpoch: 1,
                    optimizer: optimizer,
                    initializer: new Xavier(factorType: FactorType.In, magnitude: 2.34f));

                 model = new FeedForward(pnet, new List<Context> { ctx },
                    numEpoch: 1,
                    optimizer: optimizer,
                    initializer: new Xavier(factorType: FactorType.In, magnitude: 2.34f),
                    argParams: modelload.ArgParams,
                    auxParams: modelload.AuxParams
                );

            }
            catch (Exception)
            {
                // ignored
            }

            if (model == null)
            {
                model = new FeedForward(pnet, new List<Context> { ctx },
                    numEpoch: 1,
                    optimizer: optimizer,
                    initializer: new Xavier(factorType: FactorType.In, magnitude: 2.34f)
                );
            }

            model.Fit(rdtrain, rdval,
                customMetric,
                batchEndCallback: new List<BatchEndDelegate> { speed.Call },
                epochEndCallback: new List<EpochEndDelegate> { doCheckpoint.Call });

            //model.Save("checkpoint\\cnn");

            ReadData rdpredict = new ReadData("data\\train\\", batch_size, true);
            //  var testOut = model.Predict(rdpredict, 1);

            Console.WriteLine("");
        }

        private static CustomMetricResult Accuracy(SingleNArray label, SingleNArray pred, int batchSize)
        {
            int hit = 0;
            for (int i = 0; i < batchSize; i++)
            {
                var l = label[(Slice)i].Data;

                IList<int> p = new List<int>();
                for (int k = 0; k < 4; k++)
                {
                    p.Add((int)pred[(Slice)(k * batchSize + i)].Argmax());
                }

                if (l.Length == p.Count)
                {

                    var match = true;
                    for (int k = 0; k < p.Count; k++)
                    {
                        if (p[k] != (int)(l[k]))
                        {
                            match = false;
                            break;
                        }
                    }
                    if (match)
                    {
                        hit += 1;
                    }

                }

            }

            return new CustomMetricResult { SumMetric = hit, NumInst = batchSize };

        }
    }
}
