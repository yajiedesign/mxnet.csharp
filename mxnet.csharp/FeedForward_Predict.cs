using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.csharp.util;
using mxnet.numerics.single;

namespace mxnet.csharp
{
    public partial class FeedForward
    {

        public List<SingleNArray> Predict(IDataIter inputX, int? numBatch = null, bool returnData = false, bool reset = true)
        {
            if (reset)
            {
                inputX.Reset();
            }

            var dataShapes = inputX.ProvideData;
            var dataNames = dataShapes.Select(s => s.Key).ToList();
            InitPredictor(dataShapes);

            var batchSize = inputX.BatchSize;
            var dataArrays = dataNames.Select(name => this._predExec.ArgDict[name]).ToList();
            var outputList = this._predExec.Outputs.Select(s => new List<SingleNArray>()).ToList();

            List<List<SingleNArray>> dataList = null;
            List<List<SingleNArray>> labelList = null;
            if (returnData)
            {
                dataList = inputX.ProvideData.Select(s => new List<SingleNArray>()).ToList();
                labelList = inputX.ProvideLabel.Select(s => new List<SingleNArray>()).ToList();
            }

            int i = 0;
            foreach (var batch in inputX)
            {
                ExecutorManager.LoadData(batch, dataArrays);
                this._predExec.Forward(isTrain: false);
                var padded = batch.Pad;
                var realSize = batchSize - padded;

                foreach (var vitem in outputList.Zip(this._predExec.Outputs, Tuple.Create))
                {
                    vitem.Item1.Add(vitem.Item2.Slice(0, (uint)realSize).AsNumerics());

                }

                if (returnData)
                {

                    for (int j = 0; j < batch.Data.Count; j++)
                    {
                        var x = batch.Data[j];
                        dataList[j].Add(x.Slice(0, (uint)realSize).AsNumerics());
                    }

                    for (int j = 0; j < batch.Data.Count; j++)
                    {
                        var x = batch.Label[j];
                        labelList[j].Add(x.Slice(0, (uint)realSize).AsNumerics());
                    }
                }

                i += 1;
                if (numBatch != null && i == numBatch.Value)
                {
                    break;
                }

            }


            var outputs = outputList.Select(s => SingleNArray.Concatenate(0, s.ToArray())).ToList();
            if (returnData)
            {
                var data = dataList.Select(s => SingleNArray.Concatenate(0, s.ToArray()));
                var label = labelList.Select(s => SingleNArray.Concatenate(0, s.ToArray()));
            }


            return outputs;
        }

        private void InitPredictor(Dictionary<string, Shape> inputShapes)
        {
            if (_predExec != null)
            {
                var argShapes = new List<uint[]>();
                var auxShapes = new List<uint[]>();
                var outShapes = new List<uint[]>();
                this._symbol.InferShape(inputShapes, argShapes, auxShapes, outShapes);
                if (argShapes.Count == 0)
                {
                    throw new ArgumentException("Incomplete input shapes");
                }
                var predShapes = this._predExec.ArgArrays.Select(s => s.GetShape().Data()).ToList();
                if (predShapes.SequenceEqual(argShapes))
                {
                    return;
                }

            }

            var predExec = this._symbol.SimpleBind(
                this._ctx[0], inputShapes.ToDictionary(k => k.Key, v => v.Value.Data()), OpReqType.KWriteTo);
            predExec.CopyParamsFrom(this.ArgParams, this.AuxParams);

            Model.CheckArguments(this._symbol);
            this._predExec = predExec;

        }
    }
}
