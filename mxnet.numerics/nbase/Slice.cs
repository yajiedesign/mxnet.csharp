using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public class Slice
    {
        public Slice()
        {
            
        }
        public Slice(int start, int stop)
        {
            this.start = start;
            this.stop = stop;
        }

        public int start { get; set; }
        public int stop { get; set; }

        public int size => stop - start;

        public Slice Translate(uint dim)
        {
            Slice ret = new Slice();
            ret.start = start;
            if (stop == 0)
            {
                ret.stop = (int)dim;
            }else if (stop < 0)
            {
                ret.stop = (int) dim + stop;
            }
            else
            {
                ret.stop = stop;
            }

            return ret;

        }

        private static readonly Regex Reg = new Regex("(.*):(.*)");

        public static implicit operator Slice(string slice)
        {
            var m = Reg.Match(slice);
            if (m.Success)
            {
                return new Slice(int.Parse(m.Groups[1].Value), int.Parse(m.Groups[2].Value));
            }
            throw new ArgumentOutOfRangeException(nameof(slice));
        }
    }
}
