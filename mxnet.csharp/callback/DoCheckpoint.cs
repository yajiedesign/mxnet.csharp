using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.csharp.util;

namespace mxnet.csharp.callback
{
    public class DoCheckpoint
    {
        private readonly string _prefix;
        private readonly int _period;

        public DoCheckpoint(string prefix, int period = 1)
        {
            _prefix = prefix;
            _period = period;
        }

        public void Call(EpochEndParam param)
        {
            if (((param.Epoch + 1) % _period) == 0)
            {
                Model.SaveCheckpoint(_prefix, param.Epoch, param.Symbol, param.ArgParams, param.AuxParams);
            }
        }
    }
}
