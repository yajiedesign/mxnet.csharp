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
        private readonly int _batch_size;
        private readonly int _frequent;
        private bool _init;
        private readonly Stopwatch _tic;
        private int _last_count;
        private readonly ILog _log;

        public Speedometer(int batch_size, int frequent = 50, ILog log = null)
        {
            this._batch_size = batch_size;
            this._frequent = frequent;
            this._init = false;
            this._tic = new Stopwatch();
            this._last_count = 0;
            this._log = log;
            if (log == null)
            {
                this._log = LogManager.GetLogger("");
            }
        }

        public void Call(BatchEndParam param)
        {

            var count = param.nbatch;
            if (this._last_count > count)
            {
                this._init = false;
            }

            this._last_count = count;

            if (this._init)
            {
                if ((count % this._frequent) == 0)
                {
                    var speed = (double)this._frequent * this._batch_size / (this._tic.ElapsedMilliseconds / 1000f);
                    if (param.eval_metric != null)
                    {
                        var name_value = param.eval_metric.get_name_value();
                        param.eval_metric.Reset();
                        foreach (var nv in name_value)
                        {
                            _log.Info(
                                $"Epoch[{param.epoch}] Batch [{count}]\tSpeed: {speed:.00} samples/sec\tTrain-{nv.name}={nv.value}");
                        }
                    }
                    else
                    {
                        _log.Info($"Iter[{ param.epoch}] Batch [{count}]\tSpeed: {speed:.00} samples/sec");  
                    }
                    this._tic.Restart();
                }
            }
            else
            {
                this._init = true;
                this._tic.Restart();
            }

        }
    }
}
