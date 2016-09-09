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

        public List<SingleNArray> Predict(IDataIter X, int? num_batch = null, bool return_data = false, bool reset = true)
        {
            if (reset)
            {
                X.Reset();
            }

            var data_shapes = X.provide_data;
            var data_names = data_shapes.Select(s => s.Key).ToList();
            _init_predictor(data_shapes);

            var batch_size = X.batch_size;
            var data_arrays = data_names.Select(name => this._pred_exec.arg_dict[name]).ToList();
            var output_list = this._pred_exec.outputs.Select(s => new List<SingleNArray>()).ToList();

            List<List<SingleNArray>> data_list = null;
            List<List<SingleNArray>> label_list = null;
            if (return_data)
            {
                data_list = X.provide_data.Select(s => new List<SingleNArray>()).ToList();
                label_list = X.provide_label.Select(s => new List<SingleNArray>()).ToList();
            }

            int i = 0;
            foreach (var batch in X)
            {
                ExecutorManager._load_data(batch, data_arrays);
                this._pred_exec.Forward(is_train: false);
                var padded = batch.pad;
                var real_size = batch_size - padded;

                foreach (var vitem in output_list.Zip(this._pred_exec.outputs, Tuple.Create))
                {
                    vitem.Item1.Add(vitem.Item2.Slice(0, (uint)real_size).as_numerics());

                }

                if (return_data)
                {

                    for (int j = 0; j < batch.data.Count; j++)
                    {
                        var x = batch.data[j];
                        data_list[j].Add(x.Slice(0, (uint)real_size).as_numerics());
                    }

                    for (int j = 0; j < batch.data.Count; j++)
                    {
                        var x = batch.label[j];
                        label_list[j].Add(x.Slice(0, (uint)real_size).as_numerics());
                    }
                }

                i += 1;
                if (num_batch != null && i == num_batch.Value)
                {
                    break;
                }

            }


            var outputs = output_list.Select(s => SingleNArray.Concatenate(0, s.ToArray())).ToList();
            if (return_data)
            {
                var data = data_list.Select(s => SingleNArray.Concatenate(0, s.ToArray()));
                var label = label_list.Select(s => SingleNArray.Concatenate(0, s.ToArray()));
            }


            return outputs;
        }

        private void _init_predictor(Dictionary<string, Shape> input_shapes)
        {
            if (_pred_exec != null)
            {
                var arg_shapes = new List<uint[]>();
                var aux_shapes = new List<uint[]>();
                var out_shapes = new List<uint[]>();
                this._symbol.InferShape(input_shapes, arg_shapes, aux_shapes, out_shapes);
                if (arg_shapes.Count == 0)
                {
                    throw new ArgumentException("Incomplete input shapes");
                }
                var pred_shapes = this._pred_exec.arg_arrays.Select(s => s.get_shape().Data()).ToList();
                if (pred_shapes.SequenceEqual(arg_shapes))
                {
                    return;
                }

            }

            var pred_exec = this._symbol.SimpleBind(
                this._ctx[0], input_shapes.ToDictionary(k => k.Key, v => v.Value.Data()), OpReqType.KWriteTo);
            pred_exec.copy_params_from(this._arg_params, this._aux_params);

            Model._check_arguments(this._symbol);
            this._pred_exec = pred_exec;

        }
    }
}
