using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.util
{
    public delegate void EpochEndDelegate(EpochEndParam param);
    public class EpochEndParam
    {
        public int Epoch { get; }
        public Symbol Symbol { get; }
        public Dictionary<string, NdArray> ArgParams { get; }
        public Dictionary<string, NdArray> AuxParams { get; }

        public EpochEndParam(
            int epoch, 
            Symbol symbol,
            Dictionary<string, NdArray> argParams,
            Dictionary<string, NdArray> auxParams)
        {
            Epoch = epoch;
            Symbol = symbol;
            ArgParams = argParams;
            AuxParams = auxParams;
        }
    }
}
