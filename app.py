from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import UJSONResponse
import gpt_2_simple as gpt2
import tensorflow as tf
import uvicorn
import os
import gc

middleware = [Middleware(CORSMiddleware, allow_origins=['*'])]

app = Starlette(debug=False, middleware=middleware)

sess = gpt2.start_tf_sess(threads=1)
gpt2.load_gpt2(sess)


generate_count = 0


def format_prefix(history, constrain_to_topic, start_tag, end_tag):
    modified_items = [(start_tag + h.strip() + end_tag).strip() for h in history]
    # Add a period if required and newline
    modified_items = [h + (". " if h[-1] not in "?.!" else "") for h in modified_items]
    result = "".join(modified_items)
    return result


@app.route('/gpt2-chatbot', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):
    global generate_count
    global sess

    if request.method == 'GET':
        params = request.query_params
    elif request.method == 'POST':
        params = await request.json()
    elif request.method == 'HEAD':
        return UJSONResponse({'text': ''})

    start_tag = params.get('start_tag')
    end_tag = params.get('end_tag')
    temperature = params.get('temperature', 0.70)
    top_k = params.get('top_k', 0)
    top_p = params.get('top_p', 0)
    history = params.get('history', [])
    constrain_to_topic = params.get('constrain_to_topic', False)
    is_return_hint = params.get('is_return_hint', False)
    length = params.get('length', 100)

    prefix = format_prefix(history, constrain_to_topic, start_tag, end_tag)

    text_candidates = gpt2.generate(sess,
                         length=length,
                         temperature=temperature,
                         top_k=top_k,
                         top_p=top_p,
                         truncate=end_tag,
                         prefix=prefix,
                         include_prefix=False,
                         return_as_list=True
                         )
    text = text_candidates[0].replace(start_tag, '').replace(end_tag, '')
    generate_count += 1

    # If we want to predict a hint, add the most recent response from above and regenerate
    hint, hint_candidates = None, None
    if is_return_hint:
        prefix += start_tag + text.strip() + end_tag
        hint_candidates = gpt2.generate(sess,
                             length=length,
                             temperature=temperature,
                             top_k=top_k,
                             top_p=top_p,
                             truncate=end_tag,
                             prefix=prefix,
                             include_prefix=False,
                             return_as_list=True
                             )
        hint = hint_candidates[0].replace(start_tag, '').replace(end_tag, '')
        generate_count += 1

    if generate_count >= 30:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads=1)
        gpt2.load_gpt2(sess)
        generate_count = 0

    gc.collect()
    return UJSONResponse(
        {'text': text, 'hint': hint})




# IGNORE THIS
@app.route('/', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):
    global generate_count
    global sess

    if request.method == 'GET':
        params = request.query_params
    elif request.method == 'POST':
        params = await request.json()
    elif request.method == 'HEAD':
        return UJSONResponse({'text': ''})

    text = gpt2.generate(sess,
                         length=int(params.get('length', 1023)),
                         temperature=float(params.get('temperature', 0.7)),
                         top_k=int(params.get('top_k', 0)),
                         top_p=float(params.get('top_p', 0)),
                         prefix=params.get('prefix', '')[:500],
                         truncate=params.get('truncate', None),
                         include_prefix=str(params.get(
                             'include_prefix', True)).lower() == 'true',
                         return_as_list=True
                         )[0]

    generate_count += 1
    if generate_count >= 20:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads=1)
        gpt2.load_gpt2(sess)
        generate_count = 0

    gc.collect()
    return UJSONResponse({'text': text})







if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
