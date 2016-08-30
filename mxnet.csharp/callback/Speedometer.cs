using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;
using mxnet.csharp.util;

namespace mxnet.csharp.callback
{
    public class Speedometer
    {
        private int batch_size;
        private int frequent;
        private bool init;
        private Stopwatch tic;
        private int last_count;
        private ILog log;

        public Speedometer(int batch_size, int frequent = 50, ILog log = null)
        {
            this.batch_size = batch_size;
            this.frequent = frequent;
            this.init = false;
            this.tic = new Stopwatch();
            this.last_count = 0;
            this.log = log;
            if (log == null)
            {
                this.log = LogManager.GetLogger("");
            }
        }

        public void Call(BatchEndParam param)
        {

            var count = param.Nbatch;
            if (this.last_count > count)
            {
                this.init = false;
            }

            this.last_count = count;

            if (this.init)
            {
                if ((count % this.frequent) == 0)
                {
                    var speed = this.frequent * this.batch_size / (this.tic.ElapsedMilliseconds * 1000);
                    if (param.EvalMetric != null)
                    {
                        var name_value = param.EvalMetric.get_name_value();
                        param.EvalMetric.reset();
                        foreach (var nv in name_value)
                        {
                            log.Info(
                                $"Epoch[{param.Epoch}] Batch [{count}]\tSpeed: {speed:0.2} samples/sec\tTrain-{nv.Name}={nv.Value}");
                        }
                    }
                    else
                    {
                        log.Info($"Iter[{ param.Epoch}] Batch [{count}]\tSpeed: {speed:0.2} samples/sec");  
                    }
                    this.tic.Restart();
                }
            }
            else
            {
                this.init = true;
                this.tic.Restart();
            }

        }
    }
}
