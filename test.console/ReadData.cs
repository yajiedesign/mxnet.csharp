using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using mxnet.csharp;
using OpenCvSharp;

namespace test.console
{

    class DataBatch: IDataBatch
    {
        public string bucket_key { get; }

        public List<NDArray> data { get; }

        public List<NDArray> label{ get; }


        public DataBatch(List<NDArray> datas, List<NDArray> labels)
        {
            this.data = datas;
            this.label = labels;
        }

        public Dictionary<string, Shape> provide_data { get; }
        public Dictionary<string, Shape> provide_label { get; }
    }


    class ReadData : IDataIter
    {
        private readonly string _path;
        private readonly int _batchSize;

        public ReadData(string path, int batchSize)
        {
            _path = path;
            _batchSize = batchSize;

            provide_data = new Dictionary<string, Shape>
            {
                {"data", new Shape((uint) _batchSize, 3, 60, 20)}
            };

            provide_label = new Dictionary<string, Shape>
            {
                {"softmax_label", new Shape((uint) _batchSize, 4)}
            };
            var files = System.IO.Directory.EnumerateFiles(_path).ToList();

            data_s = new List<float[]>();
            label_s = new List<float[]>();
            foreach (var file in files)
            {
                float[] data = ReadFile(file);
                float[] label = ReadLabel(file);

                data_s.Add(data);
                label_s.Add(label);
            }
             rnd = new Random();
            reset();
        }

        public IEnumerator<IDataBatch> GetEnumerator()
        {
   

            var count = data_s.Count / _batchSize + 1;
            for (int batchIndex = 0; batchIndex < count; batchIndex++)
            {
                List<float> datas = new List<float>();
                List<float> labels = new List<float>();
                for (int i = 0; i < _batchSize; i++)
                {
                    var index = batchIndex * _batchSize + i;
                    if (index >= data_s.Count)
                    {
                        index = data_s.Count - 1;
                    }
    

                    datas.AddRange(data_s[index]);
                    labels.AddRange(label_s[index]);
                    //   datas.Add();
                    // labels.Add(new NDArray(label, new Shape((uint)_batchSize, 4)));

                }
                var data_all = new List<NDArray> { new NDArray(datas.ToArray(), new Shape((uint)_batchSize, 3, 60, 20)) };
                var label_all = new List<NDArray> { new NDArray(labels.ToArray(), new Shape((uint)_batchSize, 4)) };
                //   data_all.First().SetValue(3);
                // label_all.First().SetValue(3);
                yield return new DataBatch(data_all, label_all);
            }

        }

        private static readonly Regex Reg = new Regex("(\\d*)-.*", RegexOptions.Compiled);
        private List<float[]> data_s;
        private List<float[]> label_s;
        private Random rnd;

        private float[] ReadLabel(string path)
        {
            //return new float[] {9, 4, 3, 7};
            var m = Reg.Match(path);
            return m.Groups[1].Value.ToCharArray().Select(s => (float)((int)s - (int)'0')).ToArray();
        }

        private float[] ReadFile(string path)
        {
            Mat mat = new Mat(path);
            Vec3b[] array = new Vec3b[20 * 60];
            mat.GetArray(0, 0, array);


            var c1 = array.Select(s => s.Item0).Select(s => (float)(s / 255.0));
            var c2 = array.Select(s => s.Item1).Select(s => (float)(s / 255.0));
            var c3 = array.Select(s => s.Item2).Select(s => (float)(s / 255.0));


            return c1.Concat(c2).Concat(c3).ToArray();
        }

 

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public string default_bucket_key { get; set; }

        public Dictionary<string, Shape> provide_data { get; set; }
        public Dictionary<string, Shape> provide_label { get; set; }

        public int batch_size { get { return _batchSize; } }
        public void reset()
        {
      
            int[] shuffle_indices = Enumerable.Range(0, data_s.Count).OrderBy(x => rnd.Next()).ToArray();
            data_s = shuffle_indices.Select(s => data_s[s]).ToList();
            label_s = shuffle_indices.Select(s => label_s[s]).ToList();
        }
    }
}
