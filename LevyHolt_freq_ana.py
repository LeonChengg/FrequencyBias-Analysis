from wordfreq import word_frequency
import json

def compare_words_freq(s1, s2, lang="en"):
    freq1 = word_frequency(s1, lang)*1000
    freq2 = word_frequency(s2, lang)*1000
    if freq1 > freq2:
        return s1
    else:
        print(s1 + "  |  " +s2)
        return s2

def check_LH_hp(path="~/data/levyholt/levy_holt/test.txt"):
    ## check the freq of prem and hypo, see which one is more general
    with open(path, "r") as f:
        lines = f.readlines()

    count_h, count_p = 0,0
    for line in lines:
        hypo, prem, label = line.strip().split('\t')
        if label == "False":
            continue
        h_sub, h_pred, h_obj = hypo.split(",")
        p_sub, p_pred, p_obj = prem.split(",")
        
        if not (h_sub == p_sub and h_obj == p_obj): # just keep the types with same order
            #print(line)
            continue
        more_freq_pred = compare_words_freq(h_pred, p_pred)
        if more_freq_pred == h_pred:
            count_h += 1
        elif more_freq_pred == p_pred:
            count_p += 1
        else:
            print("ERRORS")
    ### print the ratio of Freq(p) < Freq(h), which represent the ration of premise is more general
    print(count_h)
    print(count_p)
    print(count_h/(count_h + count_p))


def cal_avg_PredFreq(path="~/data/test.txt"):
    with open(path, "r") as f:
        lines = f.readlines()
    pfreq, hfreq = 0, 0
    length = len(lines)
    for line in lines:
        if "EG" in path.lower():
            hypo, prem, label, lang = line.strip().split('\t')
        elif "levyholt" in path.lower():
            hypo, prem, label = line.strip().split('\t')
        h_sub, h_pred, h_obj = hypo.split(",")
        p_sub, p_pred, p_obj = prem.split(",")
        if label == "False":
            length -= 1
            continue
        if not (h_sub == p_sub and h_obj == p_obj): # just keep the types with same order
            #print(line)
            continue
        pfreq += word_frequency(p_pred, "en")*1000
        hfreq += word_frequency(h_pred, "en")*1000
    
    print("Premise Freq: " + str(100*pfreq/length))
    print("Hypothesis Freq: " + str(100*hfreq/length))

    

#cal_avg_PredFreq("data/LevyHolt/dev.txt")
cal_avg_PredFreq("/home/liang/data/data_EGEN_NS_Weeds_Local_18000/levy_holt/train.txt")