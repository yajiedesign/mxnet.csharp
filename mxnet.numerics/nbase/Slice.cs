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
            this.Start = 0;
            this.End = 0;
            this.Step = 1;

        }
        public Slice(int start, int end, int step = 1)
        {
            if (step == 0)
            {
                throw new ArgumentException($"{nameof(step)} can not be equal to 0");
            }
            //if (step > 0)
            //{
            //    if (start > 0 && end > 0 && start >= end)
            //    {
            //        throw new ArgumentException($"{nameof(start)} must be greater than {nameof(end)} when {nameof(step)} is positive");
            //    }
            //    else  if (start < 0 && end < 0 && start >= end)
            //    {
            //        throw new ArgumentException($"{nameof(start)} must be greater than {nameof(end)} when {nameof(step)} is positive");
            //    }
            //}
            //else
            //{
            //    if (start > 0 && end > 0 && end >= start)
            //    {
            //        throw new ArgumentException($"{nameof(end)} must be greater than {nameof(start)} when {nameof(step)} is negative");
            //    }
            //    else if (start < 0 && end < 0 && end >= start)
            //    {
            //        throw new ArgumentException($"{nameof(end)} must be greater than {nameof(start)} when {nameof(step)} is negative");
            //    }
            //}

            this.Start = start;
            this.End = end;
            this.Step = step;
        }

        public int Start { get; private set; }
        public int End { get; private set; }

        public int Step { get; private set; }

        public int Size
        {
            get
            {
                var temp = (int) Math.Ceiling((End - Start)/(double) Step);
                return temp > 0 ? temp : 0;
            }
        }

        /// <summary>
        /// transformed slice size absolute to relative
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        public Slice Translate(uint dim)
        {
            int retStart;
            int retEnd;
            int retStep;
            if (Step > 0)
            {
                retStart = SetStart(Start, dim , Step);
                retEnd = SetEnd(End, dim, Step);

                if (retEnd >= dim)
                {
                    retEnd = (int)dim;
                }
                retStep = Step;
            }
            else
            {
                retStart = SetEnd(Start, dim, Step);
                retEnd = SetStart(End, dim, Step);

                if (retStart >= dim)
                {
                    retStart = (int)dim - 1;
                }
                retStep = Step;
            }


            Slice ret = new Slice(retStart, retEnd, retStep);
            return ret;
        }
        private static int SetStart(int start, uint dim, int step)
        {
            if (start ==  int.MinValue)
            {
                start = 0;
            }
            else if (start ==  int.MaxValue)
            {
                start = -1;
            }
            else if (start < 0)
            {
                start = (int)dim + start;
            }

            return start;
        }

        private static int SetEnd(int end, uint dim, int step)
        {
            if (end ==  int.MaxValue)
            {
                end = (int)dim; 
            }
            else if (end == int.MinValue)
            {
                end = (int)dim;
            }
            else if (end < 0)
            {
                end = (int) dim + end;
            }
            return end;
        }

        public static implicit operator Slice(int index)
        {
            return new Slice(index, index + 1, 1);
        }


        private static readonly Regex RegNoStep = new Regex("(.*):(.*)");
        private static readonly Regex RegWithStep = new Regex("(.*):(.*):(.*)");
        public static implicit operator Slice(string slice)
        {
            int start = 0;
            if (int.TryParse(slice, out start))
            {
                return new Slice(start, start + 1, 1);
            }

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
          
            string strStart = m.Groups[1].Value;
            if (!string.IsNullOrWhiteSpace(strStart) && !int.TryParse(strStart, out start))
            {
                throw new ArgumentException($"{nameof(start)} must be number");
            }
            int end = 0;
            string strEnd = m.Groups[2].Value;
            if (!string.IsNullOrWhiteSpace(strEnd) && !int.TryParse(strEnd, out end))
            {
                throw new ArgumentException($"{nameof(end)} must be number");
            }
            int step = 1;
            if (withstep)
            {
                string strStep = m.Groups[3].Value;
                if (!string.IsNullOrWhiteSpace(strStep) && !int.TryParse(strStep, out step))
                {
                    throw new ArgumentException($"{nameof(step)} must be number");
                }
            }
            if (string.IsNullOrWhiteSpace(strStart))
            {
                start = int.MinValue;
            }
            if (string.IsNullOrWhiteSpace(strEnd))
            {
                end = int.MaxValue;
            }
            return new Slice(start, end, step);
        }

        public Slice SubSlice(Slice sub)
        {
            int localStart = 0;
            int localEnd = 0;
            int localStep = 0;
            if (this.Step > 0 && sub.Step > 0)
            {
                localStart = this.Start + sub.Start;
                localEnd = this.Start + sub.End;
                localStep = this.Step * sub.Step;
                if (localStart > this.End)
                {
                    localStart = this.End;
                }
                if (localEnd > this.End)
                {
                    localEnd = this.End;
                }
            }

            if (this.Step > 0 && sub.Step < 0)
            {
                localStart = this.Start + sub.Start;
                localEnd = this.Start + sub.End;
                localStep = this.Step * sub.Step;
                if (localStart >= this.End)
                {
                    localStart = this.End ;
                }
                if (localEnd >= this.End)
                {
                    localEnd = this.End;
                }
            }

            if (this.Step < 0 && sub.Step > 0)
            {
                localStart = this.Start - sub.Start;
                localEnd = this.Start - sub.End;
                localStep = this.Step * sub.Step;
                if (localStart < this.End)
                {
                    localStart = this.End;
                }
                if (localEnd < this.End)
                {
                    localEnd = this.End;
                }
            }


            if (this.Step < 0 && sub.Step < 0)
            {
                localEnd = this.Start - sub.End ;
                localStart = this.Start - sub.Start;
                localStep = this.Step * sub.Step;
                if (localStart <= this.End)
                {
                    localStart = this.End;
                }
                if (localEnd < this.End)
                {
                    localEnd = this.End;
                }
            }

       
      
            return new Slice(localStart, localEnd, localStep);
        }

        public static Slice[] FromShape(uint[] data)
        {
            return data.Select(s => new Slice(0, (int) s, 1)).ToArray();
        }
    }
}
