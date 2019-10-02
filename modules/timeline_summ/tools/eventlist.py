def loadEventlist():
    # get event lexicon info
    eventLexicons = set()
    eventPath = 'eventlexicon.txt'
    for line in open(eventPath, 'r', encoding='utf-8'):
        if line.rstrip() != '':
            eventLexicons.add(line.split('\t')[0])

    eventlist = list(eventLexicons)
    eventlist.sort(reverse = True)

    return eventlist
