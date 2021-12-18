$('#dataset_type').change(function () {
    var dataset_type = $(this).val();
    if(dataset_type=='WordInContext'){
        $('div.wic_field').show().prop('required',true);
    } else {
        $('div.form_field').not('.base_field').hide().prop('required',false);
    }
});
