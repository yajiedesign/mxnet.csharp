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

        public List<NDArray> Data { get; }

        public List<NDArray> Label{ get; }


        public DataBatch(List<NDArray> datas, List<NDArray> labels)
        {
            this.Data = datas;
            this.Label = labels;
        }

        public Dictionary<string, Shape> provide_data { get; }
        public Dictionary<string, Shape> provide_label { get; }
    }


    class ReadData : IDataIter
    {
        private readonly string _path;
        private readonly int _batchSize;

        public ReadData(string path,int batchSize)
        {
            _path = path;
            _batchSize = batchSize;
        }

        public IEnumerator<IDataBatch> GetEnumerator()
        {
            var files = System.IO.Directory.EnumerateFiles(_path).ToList();

            var count = files.Count/_batchSize + 1;
            for (int batchIndex = 0; batchIndex < 150; batchIndex++)
            {
                List<float> datas = new List<float>();
                List<float> labels = new List<float>();
                for (int i = 0; i < _batchSize; i++)
                {
                    var index = batchIndex * _batchSize + i;
                    if (index >= files.Count)
                    {
                        index = files.Count - 1;
                    }
                    float[] data = ReadFile(files[index]);
                    float[] label = ReadLabel(files[index]);

                    datas.AddRange(data);
                    labels.AddRange(label);
                    //   datas.Add();
                    // labels.Add(new NDArray(label, new Shape((uint)_batchSize, 4)));

                }
                var data_all = new List<NDArray> { new NDArray(datas.ToArray(), new Shape((uint)_batchSize, 3, 60, 20)) };
                var label_all = new List<NDArray> { new NDArray(labels.ToArray(), new Shape((uint)_batchSize, 4)) };

                yield return new DataBatch(data_all, label_all);
            }

        }

        private static readonly Regex Reg = new Regex("(\\d*)-.*", RegexOptions.Compiled);
        private float[] ReadLabel(string path)
        {
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

        public Dictionary<string, Shape> provide_data { get; set; } = new Dictionary<string, Shape>()
        {
            {"data", new Shape((uint) 32, 3, 60, 20)}
        };
        public Dictionary<string, Shape> provide_label { get; set; } = new Dictionary<string, Shape>()
        {
            {"softmax_label", new Shape((uint) 32, 4)}
        };

        public int batch_size { get { return _batchSize; } }
        public void reset()
        {
            
        }
    }
}
