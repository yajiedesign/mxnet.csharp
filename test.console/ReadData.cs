using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using OpenCvSharp;

namespace test.console
{

    class DataBatch
    {
        public List<float> Data { get; }
        public List<float> Label { get; }

        public DataBatch(List<float> datas, List<float> labels)
        {
            this.Data = datas;
            this.Label = labels;
        }
    }


    class ReadData : IEnumerable<DataBatch>
    {
        private readonly string _path;
        private readonly int _batchSize;

        public ReadData(string path,int batchSize)
        {
            _path = path;
            _batchSize = batchSize;
        }

        public IEnumerator<DataBatch> GetEnumerator()
        {
            var files = System.IO.Directory.EnumerateFiles(_path).ToList();

            var count = files.Count/_batchSize + 1;
            for (int batchIndex = 0; batchIndex < count; batchIndex++)
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
                    float[] data =   ReadFile(files[index]);
                    float[] label = ReadLabel(files[index]);
                    datas.AddRange(data);
                    labels.AddRange(label);

                }

                yield return new DataBatch(datas, labels);
            }

        }

        private float [] ReadLabel(string path)
        {
            Regex reg = new Regex("(\\d*)-.*", RegexOptions.Compiled);

            var m = reg.Match(path);
           return   m.Groups[1].Value.ToCharArray().Select(s => (float) ((int) s - (int) '0')).ToArray();
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
    }
}
