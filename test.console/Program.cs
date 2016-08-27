using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using mxnet.csharp;
using mxnet.csharp.optimizer;

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
                var conv1 = Symbol.Convolution(data, kernel: new Shape(5, 5), numFilter: 32);
                var pool1 = Symbol.Pooling(conv1, poolType: OperatorWarp.PoolingPoolType.Max, kernel: new Shape(2, 2),
                    stride: new Shape(1, 1));
                var relu1 = Symbol.Activation(data: pool1, actType: OperatorWarp.ActivationActType.Relu);

                var conv2 = Symbol.Convolution(relu1, kernel: new Shape(5, 5), numFilter: 32);
                var pool2 = Symbol.Pooling(data: conv2, poolType: OperatorWarp.PoolingPoolType.Avg,
                    kernel: new Shape(2, 2), stride: new Shape(1, 1));
                var relu2 = Symbol.Activation(data: pool2, actType: OperatorWarp.ActivationActType.Relu);
                var conv3 = Symbol.Convolution(data: relu2, kernel: new Shape(3, 3), numFilter: 32);
                var pool3 = Symbol.Pooling(data: conv3, poolType: OperatorWarp.PoolingPoolType.Avg,
                    kernel: new Shape(2, 2), stride: new Shape(1, 1));
                var relu3 = Symbol.Activation(data: pool3, actType: OperatorWarp.ActivationActType.Relu);
                var flatten = Symbol.Flatten(data: relu3);
                var fc1 = Symbol.FullyConnected(data: flatten, numHidden: 512);
                var fc21 = Symbol.FullyConnected(data: fc1, numHidden: 10);
                var fc22 = Symbol.FullyConnected(data: fc1, numHidden: 10);
                var fc23 = Symbol.FullyConnected(data: fc1, numHidden: 10);
                var fc24 = Symbol.FullyConnected(data: fc1, numHidden: 10);
                var fc2 = Symbol.Concat(new Symbol[] {fc21, fc22, fc23, fc24}, 4, dim: 0);
                label = Symbol.Transpose(data = label);
                label = Symbol.Reshape(data = label, shape: new Shape((uint) (batch_size*4)));
                return Symbol.SoftmaxOutput("softmax", fc2, label);
            }
        }

        static void Main(string[] args)
        {

            int batchSize = 32;
            uint W = 60;
            uint H = 20;
            float learning_rate = 1e-4f;
            float weight_decay = 1e-4f;

            ReadData rdtrain = new ReadData("data\\train\\", batchSize);
            ReadData rdval = new ReadData("data\\val\\", batchSize);


            var first = rdtrain.First();
            Context ctx = new Context(DeviceType.KGpu, 0);

            NDArray dataArray = new NDArray(new Shape((uint)batchSize, 3, W, H), ctx, false);
            NDArray labelArray = new NDArray(new Shape((uint)batchSize,4), ctx, false);

            dataArray.SyncCopyFromCPU(first.Data.ToArray());
            labelArray.SyncCopyFromCPU(first.Label.ToArray());


            dataArray.WaitToRead();
            labelArray.WaitToRead();

            //Symbol data1 = Symbol.Variable("data1");
            //Symbol data2 = Symbol.Variable("data2");
            var pnet = get_ocrnet(batchSize);


            FeedForward model = new FeedForward(pnet, new List<Context> { ctx } );

            model.Fit(rdtrain, rdval, "acc");



            Dictionary<string, NDArray> argsMap = new Dictionary<string, NDArray>();
            argsMap["data"] = dataArray;
            argsMap["softmax_label"] = labelArray;

            pnet.InferArgsMap(ctx, argsMap, argsMap);
            string error = NativeMethods.MXGetLastError();
            Optimizer opt = new Optimizer("ccsgd",learning_rate, weight_decay);
            opt.SetParam("momentum", 0.9)
                .SetParam("rescale_grad", 1.0)
                .SetParam("clip_gradient", 10);
            Executor exe = pnet.SimpleBind(ctx, argsMap);


            foreach (var item in rdtrain)
            {
                dataArray.SyncCopyFromCPU(item.Data.ToArray());
                labelArray.SyncCopyFromCPU(item.Label.ToArray());

                NDArray.WaitAll();

  
                exe.Forward(true);
                exe.Backward();
                exe.UpdateAll(opt, learning_rate, weight_decay);



            }

            Console.WriteLine(error);

        }
    }
}
