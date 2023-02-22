import io
import webbrowser
from time import monotonic


def t(t0=monotonic()):
    return monotonic() - t0


from inspect import currentframe, getframeinfo
from pathlib import Path


def debug(*args, **kwargs):
    cf = currentframe()
    filename, lineno = getframeinfo(cf.f_back).filename, cf.f_back.f_lineno
    filename = Path(filename).name
    print(f'{filename}:{lineno} [{t():.3f}]', *args, **kwargs)


import operator
import requests
import shutil
import pandas as pd
import numpy as np
from api_key import API_KEY
from pyyoutube import Api
from tqdm import tqdm
from zipfile import ZipFile


api = Api(api_key=API_KEY)


def gpu_info():
    print(f'         allocated: {torch.cuda.memory_allocated() / 2 ** 30:4.1f} Gib')
    print(f'{t():8.2f}s reserved: {torch.cuda.memory_reserved() / 2 ** 30:4.1f} Gib')


def csv_path(channel_id):
    return f'db/{channel_id}/df.csv'


def zip_path(channel_id, res):
    return f'db/{channel_id}/{res}.zip'


def hdf_path(channel_id, res):
    return f'db/{channel_id}/{res}.hdf'


def isotonic(X, Y, splits=2, increasing='auto'):
    P = np.zeros_like(Y)
    from sklearn.isotonic import IsotonicRegression
    step = 1 + len(X) // splits
    for start in range(0, len(X), step):
        end = min(len(X), start + step)
        reg = IsotonicRegression(increasing=increasing)
        reg.fit(X[start: end], Y[start: end])
        P[start: end] = reg.predict(X[start: end])
    return P


def prepare_df(channel_id, fast_mode=False, age_min=0.0, views_min=100, best_splits=3):
    df = pd.read_csv(csv_path(channel_id=channel_id), index_col=0)
    is_shorts = df.title.apply(lambda x: '#shorts' in x.lower())
    df = df[is_shorts ^ True]
    df = df[df.views >= views_min]
    df.published_at = pd.to_datetime(df.published_at)
    from sklearn.preprocessing import MinMaxScaler
    df['age'] = pd.Series(
        MinMaxScaler().fit_transform(df.published_at.astype(np.int64).to_numpy().reshape(-1, 1)).reshape(-1),
        index=df.index
    )
    df = df[df.age >= age_min]
    Y = np.log10(df.views)
    df['Y'] = Y
    if fast_mode:
        return df
    X = df.age
    from matplotlib import pyplot as plt
    from sklearn.metrics import r2_score
    plotting=False
    if plotting:
        plt.plot(X, Y, label='by_age')
    for splits in range(1, 1+3):
        P = isotonic(X, Y, splits=splits)
        print(f'{splits=} {r2_score(Y, P)=:.3f}')
        if plotting:
            plt.plot(X, P, label=f'isotonic_{splits}')
    if plotting:
        plt.legend()
        plt.show()
    df['isotonic'] = P = isotonic(X, Y, splits=best_splits)
    print(f'y1 uses {r2_score(Y, P)=:.3f}')
    Y1 = Y - P + P.mean()
    if False:
        from statistics import NormalDist
        norm = NormalDist.from_samples(Y1)
        plt.hist(Y1, bins=20, density=True)
        norm_x = np.linspace(Y1.min(), Y1.max(), 100)
        norm_y = 1 / np.sqrt(2 * np.pi * norm._sigma) * np.exp(-0.5 * (norm_x - norm._mu) ** 2 / norm._sigma)
        plt.plot(norm_x, norm_y, label='norm_pdf')
        plt.show()
    df['Y1'] = Y1
    return df


def process_thumbnails(
        channel_id='UCtqv_K5xr-OSViDaFgsFbPQ',
        res='sd',
):
    df = prepare_df(channel_id=channel_id, fast_mode=True, best_splits=1)
    with ZipFile(zip_path(channel_id=channel_id, res=res), 'r') as f_zip:
        files = set(f_zip.namelist())
        dataset = {
            video_id: name
            for video_id in df.index
            if (name := f'{video_id}_{res}.jpg') in files
               and f_zip.getinfo(name).file_size > 0
        }
        debug(f'{len(dataset)=} {len(df)=}')
        df = df[df.index.isin(dataset)]
        fname = pd.Series(dataset, name='fname')
        global torch
        import torch
        device = torch.device('cuda')
        from transformers import ViTModel, ViTImageProcessor
        from PIL import Image
        model_name = 'google/vit-base-patch16-224-in21k'
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name)
        model.to(device)
        gpu_info()
        batch_size = 64
        outputs = []
        for i in range(0, len(fname), batch_size):
            inputs = []
            for e in fname[i: i+batch_size]:
                with f_zip.open(e) as f:
                    x = Image.open(f)
                    np.array(x)  # SIDE EFFECTS!
                    inputs.append(x)
            inputs: torch.Tensor = image_processor.preprocess(inputs, return_tensors='pt').pixel_values
            if i == 0:
                debug(inputs.shape)
            inputs = inputs.to(device)
            output = model(inputs).pooler_output.detach().cpu().numpy()
            outputs.append(output)
            if i == 0:
                debug(output.shape)
                gpu_info()
        outputs = np.concatenate(outputs)
        debug(outputs.shape)
        df = pd.DataFrame(outputs, index=fname)
        df.to_hdf(hdf_path(channel_id=channel_id, res=res), key='df')
        gpu_info()


def dump_thumbnails(
        channel_id='UCtqv_K5xr-OSViDaFgsFbPQ',
        res='sd',
):
    df = pd.read_csv(csv_path(channel_id=channel_id), index_col=0)
    print(df.describe())
    with ZipFile(zip_path(channel_id=channel_id, res=res), 'a') as f_zip:
        files = set(f_zip.namelist())
        def thumbnail_url(video_id, res='sd'):
            assert res in ('', 'mq', 'hq', 'sd', 'maxres')
            return f'https://i.ytimg.com/vi/{video_id}/{res}default.jpg'
        to_download = [
            (thumbnail_url(video_id, res), video_id, name)
            for video_id in df.index
            if (name := f'{video_id}_{res}.jpg') not in files
        ]
        errors = []
        for url, video_id, name in (pbar := tqdm(to_download)):
            r = requests.get(url, stream=True)
            if r.status_code == 404:
                errors.append(video_id)
                pbar.set_description(f'{len(errors)=}')
                continue
            r.raise_for_status()
            r.raw.decode_content = True
            with f_zip.open(name, 'w') as f:
                shutil.copyfileobj(r.raw, f)
        print(errors)


def dump_channel_csv(
        channel_id='UCtqv_K5xr-OSViDaFgsFbPQ',  # AlinaRin
        hl='ru',
        count=None,
        limit=50,
):
    info = api.get_channel_info(channel_id=channel_id, hl=hl)
    # {viewCount: 129573928, videoCount: 1442}
    uploads_id = info.items[0].contentDetails.relatedPlaylists.uploads
    info = api.get_playlist_items(playlist_id=uploads_id, parts='contentDetails', count=count, limit=limit)
    video_ids = [e.contentDetails.videoId for e in info.items]
    data = []
    for i in tqdm(range(0, len(video_ids), limit)):
        chunk = video_ids[i: min(len(video_ids), i + limit)]
        info = api.get_video_by_id(video_id=chunk, parts='snippet,statistics', hl=hl)
        data.extend([
            (video.id, video.statistics.likeCount, video.statistics.commentCount, video.statistics.viewCount,
             video.snippet.publishedAt, video.snippet.title)
            for video in info.items])
    df = pd.DataFrame(data, columns=['id', 'likes', 'comments', 'views', 'published_at', 'title'])
    df.to_csv(csv_path(channel_id=channel_id), index=False)
    print(df)


def process_ml(
    channel_id='UCtqv_K5xr-OSViDaFgsFbPQ',
    res='sd',
    age_min=0.0,
    use_y1=True,
    best_splits=1,
):
    df = prepare_df(channel_id=channel_id, fast_mode=False, age_min=age_min, best_splits=best_splits)
    df_ml = pd.read_hdf(hdf_path(channel_id=channel_id, res=res), key='df')
    features = df_ml.columns
    df['fname'] = df.index.map(lambda x: f'{x}_{res}.jpg')
    df = df.join(df_ml, on='fname', how='inner')
    X, Y = df[features].values, (df.Y1 if use_y1 else df.Y).values
    from sklearn.decomposition import PCA
    # X = PCA(16).fit_transform(X)
    from sklearn.linear_model import RidgeCV
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
    # reg = RidgeCV(alphas=np.logspace(5, -5, 5--5+1))
    reg = SVR(C=1e-1, epsilon=1e-1)
    n_fold = 5
    print(score := cross_val_score(reg, X, Y, cv=KFold(n_splits=n_fold, shuffle=True)), f'{np.mean(score):.1%}')
    p = cross_val_predict(reg, X, Y, cv=KFold(n_splits=n_fold, shuffle=True))
    if False:
        from sklearn.metrics import PredictionErrorDisplay
        display = PredictionErrorDisplay(y_true=Y, y_pred=p)
        display.plot(kind='actual_vs_predicted')
        from matplotlib import pyplot as plt
        plt.show()
    def print_r2(x):
        return f'{x:.2f}'
    if False:
        from matplotlib import pyplot as plt
        plt.plot(df.age, df.isotonic, label='isotonic')
        plt.plot(df.age, p, label='svr')
        plt.legend()
        plt.show()
    data = list(zip(range(len(Y)),
                    df.fname.values,
                    map(print_r2, Y),
                    map(print_r2, p),
                    df.views.values,
                    map(print_r2, df.age.values),
                    map(print_r2, df.Y1.values),
                    ))
    import PySimpleGUI as sg
    kTable = '-TABLE-'
    window = sg.Window('Files', layout=[
        [sg.InputText('status', expand_x=True, readonly=True, key='-S-')],
        [sg.InputText('title', expand_x=True, readonly=True, key='-T-')],
        [sg.B('Go Youtube!', key='-G-')],
        [sg.Table(data, headings=['id', 'fname', 'Y', 'P', 'views', 'age', 'Y1'],
                  num_rows=50, key=kTable, enable_events=True, enable_click_events=True),
         sg.Image(size=(640, 640), key='-I-')],
    ])
    f_sort = None
    def table_row_selected(row):
        from PIL import Image
        assert 0 <= row < len(data)
        def read_image(fname):
            with ZipFile(zip_path(channel_id=channel_id, res=res), 'r') as f_zip:
                img = Image.open(f_zip.open(fname))
            bio = io.BytesIO()
            img.save(bio, format='PNG')
            return bio.getvalue()
        window['-I-'].update(data=read_image(data[row][1]))
        id = data[row][0]
        window['-T-'].update(df.title[id])
        window['-S-'].update(' '.join(f'{c}={df[c].iloc[id]}' for c in ['likes', 'comments', 'views', 'published_at']))
    row = None
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        if isinstance(event, tuple):
            if event[0] == kTable and event[1] == '+CLICKED+':
                if event[2][0] == -1:
                    col = event[2][1]
                    if f_sort == col:
                        data.reverse()
                    else:
                        data = sorted(data, key=operator.itemgetter(col))
                        f_sort = col
                    window[kTable].update(data)
                else:
                    table_row_selected(row := event[2][0])
                continue
        elif event == '-G-' and row is not None:
            webbrowser.open(f'https://www.youtube.com/watch?v={df.index[data[row][0]]}')
        elif event == kTable:
            if len(e := values.get(kTable)) == 1:
                table_row_selected(row := e[0])
        else:
            debug(event, values)


if __name__ == '__main__':
    # dump_channel_csv()
    # dump_thumbnails()
    # process_thumbnails()
    process_ml(age_min=0.35)
