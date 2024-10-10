// Interactive Bar Plot
var barData = [{
    x: ['Urban', 'Suburban', 'Rural'],
    y: [50, 30, 20],
    type: 'bar'
}];

Plotly.newPlot('chart', barData);

// Interactive Scatter Plot
var scatterTrace = {
    x: [40000, 50000, 60000, 70000, 80000],
    y: [2000, 3000, 4000, 5000, 6000],
    mode: 'markers',
    type: 'scatter'
};

var scatterData = [scatterTrace];

Plotly.newPlot('scatter', scatterData);
