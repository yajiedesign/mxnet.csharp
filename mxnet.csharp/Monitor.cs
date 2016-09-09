using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public class Monitor
    {
        private readonly Func<NdArray, NdArray> _statFunc;
        private int _interval;
        private readonly bool _activated;
        private readonly IList<Tuple<int, string, NdArray>> _queue;
        private readonly int _step;
        private readonly IList<Executor> _exes;
        private readonly Regex _reProg;
        private bool _sort;


        NdArray asum_stat(NdArray x)
        {
            //TODO  return ndarray.norm(x)/Math.Sqrt(x.size());
            return x;
        }

        public void StatHelper(string name, IntPtr arrayPtr, IntPtr param2)
        {
            if (!this._activated || !this._reProg.IsMatch(name))
            {
                return;
            }

            var array = new NdArray(arrayPtr, writable: false);
            this._queue.Add(Tuple.Create(this._step, name, this._statFunc(array)));
        }
        public Monitor(int interval, Func<NdArray, NdArray> statFunc = null, string pattern = ".*", bool sort = false)
        {
            if (statFunc == null)
            {
                statFunc = asum_stat;
            }
            this._statFunc = statFunc;
            this._interval = interval;
            this._activated = false;
            this._queue = new List<Tuple<int, string, NdArray>>();
            this._step = 0;
            this._exes = new List<Executor>();
            this._reProg = new Regex(pattern);
            this._sort = sort;



        }


        public void Install(Executor exe)
        {
            exe.SetMonitorCallback(StatHelper);
            this._exes.Add(exe);
        }

        public void Tic()
        {
            //TODO tic
        }

        public void Toc()
        {
            //TODO toc
        }

        public void TocPrint()
        {
            //TODO TocPrint
        }
    }
}
