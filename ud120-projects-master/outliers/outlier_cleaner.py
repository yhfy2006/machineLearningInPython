#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    list = []
    for ii in range(0,len(predictions)):
        list.append((ages[ii],net_worths[ii],[abs(net_worths[ii][0]-predictions[ii][0])]))

    list.sort(key=lambda tup: tup[2], reverse=True)
    
    cleaned_data = list[10:]

    ### your code goes here

    
    return cleaned_data

