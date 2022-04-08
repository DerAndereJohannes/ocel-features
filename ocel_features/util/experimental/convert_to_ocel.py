import pm4py
import ocel

def xes_to_ocel():
    log = {'ocel:objects': {},
           'ocel:events': {},
           'ocel:global-log': {'ocel:version': '0.1',
                               'ocel:ordering': 'timestamp',
                               'ocel:attribute-names': set(),
                               'ocel:object-types': {'Offer', 'Application', 'Resource'}}
        }

    offer_xes = pm4py.read_xes('bpi/BPI Challenge 2017 - Offer log.xes')

    for case in offer_xes:
        o_obj = case.__dict__['_attributes']
        o_id = o_obj['concept:name']

        new_object = {'ocel:type': 'Offer', 'ocel:ovmap': {k: v for k, v in o_obj.items() if k not in {'concept:name'}}}

        log['ocel:objects'][o_id] = new_object


    xes = pm4py.read_xes('bpi/BPI Challenge 2017 - Application Log.xes')

    e_number = 0

    for case in xes:
        a_obj = case.__dict__['_attributes']
        a_id = a_obj['concept:name']

        new_object = {'ocel:type': 'Application', 'ocel:ovmap': {k: v for k, v in a_obj.items() if k not in {'concept:name'}}}

        log['ocel:objects'][a_id] = new_object


        for e in case:
            new_event = {'ocel:activity': e['concept:name'],
                         'ocel:timestamp': e['time:timestamp'],
                         'ocel:omap': {a_id, e['org:resource']},
                         'ocel:vmap': {k: att for k, att in e.items() if k not in {'EventID', 'EventOrigin', 'org:resource'}}
                    }

            if 'O_' in e['concept:name']:
                if 'OfferID' not in e:
                    new_event['ocel:omap'].add(e['EventID'])
                else:
                    new_event['ocel:omap'].add(e['OfferID'])

            elif 'W_' in e['concept:name']:
                work_item = e['EventID']
                if work_item not in log['ocel:objects']:
                    new_object = {'ocel:type': 'Workflow', 'ocel:ovmap': {}}
                    log['ocel:objects'][work_item] = new_object

                new_event['ocel:omap'].add(e['EventID'])

            log['ocel:events'][f'e{e_number}'] = new_event

            if 'org:resource' in e and e['org:resource'] not in log['ocel:objects']:
                new_resource = {'ocel:type': 'Resource', 'ocel:ovmap': {}}
                log['ocel:objects'][e['org:resource']] = new_resource

            e_number = e_number + 1


    return log




