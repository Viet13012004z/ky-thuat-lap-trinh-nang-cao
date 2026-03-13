from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def load_and_clean_data():
    df = pd.read_csv('data.csv')
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

@app.route('/')
def index():
    df = load_and_clean_data()
    # Lấy 4 sản phẩm có rating cao nhất để hiển thị ở trang chủ
    top_rated = df.sort_values(by='rating', ascending=False).head(4).to_dict('records')
    return render_template('index.html', top_rated=top_rated)

@app.route('/search')
def search():
    query = request.args.get('q', '').lower()
    df = load_and_clean_data()
    results = df[df['item_name'].str.lower().str.contains(query)].to_dict('records')
    suggestions = []
    if results:
        main = results[0]
        suggestions = df[(df['style_code'] == main['style_code']) & 
                         (df['category'] != main['category']) & 
                         (df['gender'] == main['gender'])].head(3).to_dict('records')
    return render_template('search_result.html', results=results, suggestions=suggestions, query=query)

@app.route('/recommend', methods=['POST'])
def recommend():
    u_name = request.form.get('name')
    u_age = float(request.form.get('age'))
    u_gender = int(request.form.get('gender'))
    u_style = int(request.form.get('style'))
    
    df = load_and_clean_data()
    f_df = df[df['gender'] == u_gender].copy()
    if f_df.empty: return render_template('result.html', recs=[], name=u_name)

    scaler = MinMaxScaler()
    X = f_df[['age_min', 'style_code']].values.astype(float)
    X[:, 1] *= 5
    scaled_X = scaler.fit_transform(X)
    scaled_user = scaler.transform(np.array([[u_age, u_style * 5]]))

    model = NearestNeighbors(n_neighbors=min(6, len(f_df)), metric='manhattan')
    model.fit(scaled_X)
    distances, indices = model.kneighbors(scaled_user)
    
    recs = []
    for idx, dist in zip(indices.flatten(), distances.flatten()):
        item = f_df.iloc[idx]
        recs.append({
            'name': item['item_name'], 
            'price': "{:,}đ".format(int(item['price'])).replace(',', '.'),
            'score': round(max(75.0, 100 - (dist * 10)), 1),
            'image': item['image_url']
        })
    return render_template('result.html', name=u_name, recs=recs, age=int(u_age), gender=u_gender, style=u_style)

@app.route('/checkout')
def checkout():
    item_name = request.args.get('item_name')
    image = request.args.get('image')
    return render_template('checkout.html', item_name=item_name, image=image)

@app.route('/rate', methods=['POST'])
def rate():
    item_name = request.form.get('item_name')
    user_rating = float(request.form.get('rating'))
    df = load_and_clean_data()
    if item_name in df['item_name'].values:
        idx = df[df['item_name'] == item_name].index[0]
        old_r = df.at[idx, 'rating'] if not pd.isna(df.at[idx, 'rating']) else 0.0
        old_c = int(df.at[idx, 'rating_count']) if not pd.isna(df.at[idx, 'rating_count']) else 0
        new_c = old_c + 1
        new_r = round(((old_r * old_c) + user_rating) / new_c, 1)
        df.at[idx, 'rating'] = new_r
        df.at[idx, 'rating_count'] = new_c
        df.to_csv('data.csv', index=False)
        return jsonify({"status": "success", "new_rating": new_r})
    return jsonify({"status": "error"}), 404

if __name__ == '__main__':
    app.run(debug=True)