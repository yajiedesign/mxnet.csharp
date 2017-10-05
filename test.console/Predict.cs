using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.csharp;
using mxnet.csharp.initializer;
using mxnet.csharp.optimizer;

namespace test.console
{
    class Predict
    {

        public void DoPredict()
        {
            int batch_size = 16;
            Context ctx = new Context(DeviceType.KCpu, 0);
            Optimizer optimizer = new CcSgd(momentum: 0.9f, learningRate: 0.001f, wd: 0.00001f, rescaleGrad: 1.0f / batch_size);

            var modelload = FeedForward.Load("checkpoint\\tag", ctx: ctx,
                numEpoch: 1,
                optimizer: optimizer,
                initializer: new Xavier(factorType: FactorType.In, magnitude: 2.34f));



        }
    }
}
