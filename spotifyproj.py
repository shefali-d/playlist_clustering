import pandas as pd 
import numpy as np
import spotipy 
sp = spotipy.Spotify() 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from spotipy.oauth2 import SpotifyClientCredentials 
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from sklearn.decomposition import PCA
from kneed import KneeLocator
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math

cid = "4fd6aab33a6d4691a1e0238c8c0159ac" 
secret = "" 


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                               client_secret=secret,
                                               redirect_uri="http://localhost/?code=",
                                               scope="user-library-read"))

results = sp.current_user_saved_tracks()
col_names = ["playlist_name", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]


@st.cache
def get_features(playlist_id):
    songs = sp.playlist_tracks(playlist_id)
    num_songs, dance, energy, loud, speechiness, accousticness, instrumentalness, liveness, valence, tempo = 0,0,0,0,0,0,0,0,0,0
    for i, item in enumerate(songs['items']):
        num_songs += 1
        track = item['track']
        features = sp.audio_features(track['id']) 
        for i, item in enumerate(features):
            dance += item['danceability']
            energy += item['energy']
            loud += item['loudness']
            speechiness += item['speechiness']
            accousticness += item['acousticness']
            instrumentalness += item['instrumentalness']
            liveness += item['liveness']
            valence += item['valence']
            tempo += item['tempo']
    dance = dance/num_songs
    energy = energy/num_songs
    loud = loud/num_songs
    speechiness = speechiness/num_songs
    accousticness = accousticness/num_songs
    instrumentalness = instrumentalness/num_songs
    liveness = liveness/num_songs
    valence = valence/num_songs
    tempo = tempo/num_songs
    return(dance, energy, loud, speechiness, accousticness, instrumentalness, liveness, valence, tempo)

@st.cache
def get_playlists(data):
    playlists = sp.current_user_playlists()
    for i, item in enumerate(playlists['items']):
        #print("%d %s" % (i, item['name']))
        curr_playlist_name = item['name']
        curr_playlist_id = item['id']
        dance, energy, loud, speechiness, accousticness, instrumentalness, liveness, valence, tempo = get_features(curr_playlist_id)
        new_data = pd.DataFrame(data = [[curr_playlist_name, dance, energy, loud, speechiness, accousticness, instrumentalness, liveness, valence, tempo]], columns = col_names)
        data = data.append(new_data)
    return data

@st.cache
def calculate_wcss(data, kmax):
    wcss = []
    for n in range(2, kmax):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    
    return wcss

@st.cache
def optimal_number_of_clusters(wcss,playlist_count):
    x1, y1 = 2, wcss[0]
    x2, y2 = playlist_count, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


def clustering(data):
    features = data.loc[:, data.columns != 'playlist_name']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    playlist_count = scaled_features.shape[0]
    y = calculate_wcss(scaled_features, playlist_count)
    num_cluster = optimal_number_of_clusters(y,playlist_count)
    kmeans_model = KMeans(n_clusters = num_cluster)

    kmeans_model.fit(scaled_features)

    labels = kmeans_model.labels_
    cluster_features = kmeans_model.cluster_centers_
    # RADAR CHARTS
    categories = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    fig = go.Figure()

    for i in range(num_cluster):

        fig.add_trace(go.Scatterpolar(
            r = cluster_features[i],
            theta = categories,
            fill = 'toself',
            name = i
        ))
    st.write(fig)


    cluster = st.selectbox(label = "Choose a cluster to see the playlists for ", options = range(0,num_cluster))
    st.write("cluster " + str(cluster) + ": ")
    res_list = [i for i in range(len(labels)) if labels[i] == cluster]
    for y in res_list:
        st.write(data.iloc[y,0])

#@st.cache
def display():
    # create data frame
    data = pd.DataFrame(columns = col_names)
    data = get_playlists(data)


    st.title("analyze your spotify playlists!")
    st.write("audio features of your playlists:")
    st.write(data)

    y_axis = st.selectbox(label = "Choose an audio feature to visualize", options = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"])
    chart_data = pd.DataFrame({"playlist name": data['playlist_name'], y_axis: data[y_axis]})
    graph1 = alt.Chart(chart_data).mark_bar().encode(x = "playlist name", y = y_axis, color = y_axis)
    st.altair_chart(graph1)


    x1 = st.selectbox(label = "an audio feature for the x-axis", options = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"], index = 1)
    y1 = st.selectbox(label = "an audio feature for the y-axis", options = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"], index = 2)
    chart_data2 = pd.DataFrame({"playlist name": data['playlist_name'], x1: data[x1], y1: data[y1]})
    graph2 = alt.Chart(chart_data2).mark_circle(size=60).encode(x = x1, y = y1, color = "playlist name", tooltip = ["playlist name", x1, y1]).interactive()
    st.altair_chart(graph2)

    graph2 = alt.Chart(chart_data2)

    clustering(data)

display()







