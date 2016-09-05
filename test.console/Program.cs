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
using mxnet.numerics.nbase;
using mxnet.numerics.single;
using Shape = mxnet.csharp.Shape;

namespace test.console
{
    class Program
    {

        static Symbol get_ocrnet(int batch_size)
        {
            using (NameScop scop = new NameScop())
            {
                var data = Symbol.Variable("data");
                var label = Symbol.Variable("softmax_label");
                var conv1 = Symbol.Convolution(data, kernel: new Shape(5, 5), num_filter: 32);
                var pool1 = Symbol.Pooling(conv1, pool_type: Symbol.PoolingPoolType.Max, kernel: new Shape(2, 2),
                    stride: new Shape(1, 1));
                var relu1 = Symbol.Activation(data: pool1, act_type: Symbol.ActivationActType.Relu);

                var conv2 = Symbol.Convolution(relu1, kernel: new Shape(5, 5), num_filter: 32);
                var pool2 = Symbol.Pooling(data: conv2, pool_type: Symbol.PoolingPoolType.Avg,
                    kernel: new Shape(2, 2), stride: new Shape(1, 1));
                var relu2 = Symbol.Activation(data: pool2, act_type: Symbol.ActivationActType.Relu);
                var conv3 = Symbol.Convolution(data: relu2, kernel: new Shape(3, 3), num_filter: 32);
                var pool3 = Symbol.Pooling(data: conv3, pool_type: Symbol.PoolingPoolType.Avg,
                    kernel: new Shape(2, 2), stride: new Shape(1, 1));
                var relu3 = Symbol.Activation(data: pool3, act_type: Symbol.ActivationActType.Relu);
                var flatten = Symbol.Flatten(data: relu3);
                var fc1 = Symbol.FullyConnected(data: flatten, num_hidden: 512);
                var fc21 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc22 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc23 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc24 = Symbol.FullyConnected(data: fc1, num_hidden: 10);
                var fc2 = Symbol.Concat(new Symbol[] { fc21, fc22, fc23, fc24 }, 4, dim: 0);
                label = Symbol.Transpose(data = label);
                label = Symbol.Reshape(data = label, shape: new Shape((uint)(batch_size * 4)));
                return Symbol.SoftmaxOutput("softmax", fc2, label);
            }
        }

        static void Main(string[] args)
        {

            NumericsTest test = new NumericsTest();
            test.Test();
            return;
            var log4_net_config = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), "log4net.config");
            XmlConfigurator.Configure(new FileInfo(log4_net_config));


            int batch_size = 32;
            uint w = 60;
            uint h = 20;
            float learning_rate = 1e-4f;
            float weight_decay = 1e-4f;

            ReadData rdtrain = new ReadData("data\\train\\", batch_size);
            ReadData rdval = new ReadData("data\\val\\", batch_size);


            //var first = rdtrain.First();
            Context ctx = new Context(DeviceType.KGpu, 0);

            //NDArray dataArray = new NDArray(new Shape((uint)batchSize, 3, W, H), ctx, false);
            //NDArray labelArray = new NDArray(new Shape((uint)batchSize,4), ctx, false);


            //Symbol data1 = Symbol.Variable("data1");
            //Symbol data2 = Symbol.Variable("data2");
            var pnet = get_ocrnet(batch_size);
            Speedometer speed = new Speedometer(batch_size, 50);

            CustomMetric custom_metric = new CustomMetric((l, p) => Accuracy(l, p, batch_size));

            Optimizer optimizer = new CcSgd(momentum: 0.9f, learning_rate: 0.001f, wd: 0.00001f, rescale_grad: 1.0f / batch_size);


            FeedForward model = new FeedForward(pnet, new List<Context> { ctx },
                num_epoch: 10,
                optimizer: optimizer,
                initializer: new Xavier(factor_type: FactorType.In, magnitude: 2.34f)

                );


            model.Fit(rdtrain, rdval,
                custom_metric,
                batch_end_callback: new List<Action<mxnet.csharp.util.BatchEndParam>> { speed.Call });
            Console.WriteLine("");

        }

        private static CustomMetricResult Accuracy(SingleNArray label, SingleNArray pred, int batch_size)
        {
            int hit = 0;
            for (int i = 0; i < batch_size; i++)
            {
                var l = label[i];

                List<int> p = new List<int>();
                for (int k = 0; k < 4; k++)
                {
                    p.Add((int)pred[k * batch_size + i].Argmax());
                }

                if (l.shape.size == p.Count)
                {

                    var match = true;
                    for (int k = 0; k < p.Count; k++)
                    {
                        if (p[k] != (int)(l.data[k]))
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

            return new CustomMetricResult { sum_metric = hit, num_inst = batch_size };

        }
    }
}
