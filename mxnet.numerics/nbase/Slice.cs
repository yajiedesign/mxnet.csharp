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
            this.start = 0;
            this.end = 0;
            this.step = 1;

        }
        public Slice(int start, int end, int step = 1)
        {
            if (step == 0)
            {
                throw new ArgumentException($"{nameof(step)} can not be equal to 0");
            }
            if (step > 0)
            {
                if (start > 0 && end > 0 && start >= end)
                {
                    throw new ArgumentException($"{nameof(start)} must be greater than {nameof(end)} when {nameof(step)} is positive");
                }
                else  if (start < 0 && end < 0 && start >= end)
                {
                    throw new ArgumentException($"{nameof(start)} must be greater than {nameof(end)} when {nameof(step)} is positive");
                }
            }
            else
            {
                if (start > 0 && end > 0 && end >= start)
                {
                    throw new ArgumentException($"{nameof(end)} must be greater than {nameof(start)} when {nameof(step)} is negative");
                }
                else if (start < 0 && end < 0 && end >= start)
                {
                    throw new ArgumentException($"{nameof(end)} must be greater than {nameof(start)} when {nameof(step)} is negative");
                }
            }

            this.start = start;
            this.end = end;
            this.step = step;
        }

        public int start { get; private set; }
        public int end { get; private set; }

        public int step { get; private set; } 

        public int size => (end - start) / step;
        /// <summary>
        /// transformed slice size absolute to relative
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        public Slice Translate(uint dim)
        {
            int ret_start;
            int ret_end;
            int ret_step;
            if (step > 0)
            {
                ret_start = SetStart(start, dim);
                ret_end = SetEnd(end, dim);
                ret_step = step;
            }
            else
            {
                ret_start = SetEnd(start, dim);
                ret_end = SetStart(end, dim);
                ret_step = step;
            }
            Slice ret = new Slice(ret_start, ret_end, ret_step);
            return ret;
        }
        private static int SetStart(int start, uint dim)
        {
            if (start < 0)
            {
                start = (int)dim + start;
            }
            return start;
        }
        private static int SetEnd( int end, uint dim)
        {
            if (end == 0)
            {
                end = (int)dim;
            }
            else if (end < 0)
            {
                end = (int)dim + end;
            }
            return end;
        }



        private static readonly Regex RegNoStep = new Regex("(.*):(.*)");
        private static readonly Regex RegWithStep = new Regex("(.*):(.*):(.*)");
        public static implicit operator Slice(string slice)
        {
            bool withstep = false;
            var m = RegWithStep.Match(slice);
            if (!m.Success)
            {
                m = RegNoStep.Match(slice);
            }
            else
            {
                withstep = true;
            }

            if (!m.Success) { throw new ArgumentException(nameof(slice));}
            int start = 0;
            string str_start = m.Groups[1].Value;
            if (!string.IsNullOrWhiteSpace(str_start) && !int.TryParse(str_start, out start))
            {
                throw new ArgumentException($"{nameof(start)} must be number");
            }
            int end = 0;
            string str_end = m.Groups[2].Value;
            if (!string.IsNullOrWhiteSpace(str_end) &&!int.TryParse(str_end, out end))
            {
                throw new ArgumentException($"{nameof(end)} must be number");
            }

            int step = 1;
            if (withstep)
            {
                string str_step = m.Groups[3].Value;
                if (!string.IsNullOrWhiteSpace(str_step) && !int.TryParse(str_step, out step))
                {
                    throw new ArgumentException($"{nameof(step)} must be number");
                }
            }
            return new Slice(start, end, step);
        }
    }
}
