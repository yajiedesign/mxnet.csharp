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
        private readonly int _batchSize;
        private readonly int _frequent;
        private bool _init;
        private readonly Stopwatch _tic;
        private int _lastCount;
        private readonly ILog _log;

        public Speedometer(int batchSize, int frequent = 50, ILog log = null)
        {
            this._batchSize = batchSize;
            this._frequent = frequent;
            this._init = false;
            this._tic = new Stopwatch();
            this._lastCount = 0;
            this._log = log;
            if (log == null)
            {
                this._log = LogManager.GetLogger("");
            }
        }

        public void Call(BatchEndParam param)
        {

            var count = param.Nbatch;
            if (this._lastCount > count)
            {
                this._init = false;
            }

            this._lastCount = count;

            if (this._init)
            {
                if ((count % this._frequent) == 0)
                {
                    var speed = (double)this._frequent * this._batchSize / (this._tic.ElapsedMilliseconds / 1000f);
                    if (param.EvalMetric != null)
                    {
                        var nameValue = param.EvalMetric.get_name_value();
                        param.EvalMetric.Reset();
                        foreach (var nv in nameValue)
                        {
                            _log.Info(
                                $"Epoch[{param.Epoch}] Batch [{count}]\tSpeed: {speed:.00} samples/sec\tTrain-{nv.Name}={nv.Value}");
                        }
                    }
                    else
                    {
                        _log.Info($"Iter[{ param.Epoch}] Batch [{count}]\tSpeed: {speed:.00} samples/sec");  
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
